from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,)
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from pinecone_text.sparse import BM25Encoder
import pandas as pd
import openai
from langchain.chains.llm import LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers import PineconeHybridSearchRetriever
import os
from langchain.chat_models import ChatOpenAI
from item import Item
from notifier import Notifier
from streaming_callback import StreamingCallbackHandler
import logging
import re
import sqlite3
import ast


class PineconeSearch:

    def __init__(self, notifier: Notifier, logger):
        self.notifier: Notifier = notifier
        self.logger = logger
        # pinecone
        pinecone.init(
            api_key = 'xxx',
            environment = 'asia-southeast1-gcp'
        )
        self.index_name = "xxx"
        self.index_name_hotplc = "yyy"
        

        self.index = pinecone.Index(self.index_name)
        self.index_hotplc = pinecone.Index(self.index_name_hotplc)

        # openAI embedding
        self.model_name = 'text-embedding-ada-002'
        openai.api_key = 'xxx'
        self.embed = OpenAIEmbeddings(
            model=self.model_name,
            openai_api_key= 'yyy')

        # 하나카드 외국인 데이터
        self.df_hana = pd.read_excel('./외국인(추출)_1023.xlsx')

        # BM25
        self.bm25 = BM25Encoder()

        # BM25 fit 데이터 
        self.df = pd.read_excel('./FIT_0728.xlsx',dtype=str)

        self.bm25.fit(self.df['New_Column3'])

        os.environ["OPENAI_API_KEY"] = "xxx"
    
        self.retriever = PineconeHybridSearchRetriever(
            embeddings=self.embed, sparse_encoder=self.bm25, index=self.index, top_k= int(5), alpha= float(0.7))

        self.retriever_hotplc = PineconeHybridSearchRetriever(
            embeddings=self.embed, sparse_encoder=self.bm25, index=self.index_hotplc, top_k= int(5), alpha= float(0.7))

        self.general_system_template = self.read_prompt_db()

        print("prompt : ", self.general_system_template)

        # self.general_system_template = r""" 
        # Respond in Korean. Responding in English. You can only print up to 5 responses. All responses are printed in HTML format.You can only print up to 5 responses. All responses are printed in HTML format.In your response, summarize your store'\''s description and reviews in 150 characters or less to showcase each store'\''s features. Use emojis in your responses to illustrate your store'\''s features.\nIf you can'\''t include a store review in your response, then replace it with a short store description of 100 characters or less. If the store you'\''re looking for doesn'\''t exist in the source, clearly state that it doesn'\''t exist. if the question asks for '\''more'\'', find and print the next result, excluding the previous one. {context}

        # """
        self.llm = ChatOpenAI(model_name = "gpt-3.5-turbo-0613", temperature=0)
        self.streaming_llm = ChatOpenAI(model_name = "gpt-3.5-turbo-0613",streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0)

        self.general_user_template = "Question:```{question}```"
        self.messages = [
                    SystemMessagePromptTemplate.from_template(self.general_system_template),
                    HumanMessagePromptTemplate.from_template(self.general_user_template)
        ]
        self.qa_prompt = ChatPromptTemplate.from_messages(self.messages )

        self.question_generator = LLMChain(llm=self.llm, prompt=CONDENSE_QUESTION_PROMPT)
        # doc_chain = load_qa_chain(llm, chain_type="stuff", prompt=qa_prompt)
        self.doc_chain = load_qa_chain(self.streaming_llm, chain_type="stuff", prompt=self.qa_prompt)

        self.qa = ConversationalRetrievalChain(
                    retriever=self.retriever, combine_docs_chain=self.doc_chain, question_generator=self.question_generator,
                    return_source_documents=True)
  
        self.qa_hotplc = ConversationalRetrievalChain(
                    retriever=self.retriever_hotplc, combine_docs_chain=self.doc_chain, question_generator=self.question_generator,
                    return_source_documents=True)

        self.chat_history = []
        self.logger.debug("Search instance is initialized")

    def read_prompt_db(self):
        con = sqlite3.connect("./pine.sqlite")
        cur = con.cursor()
        cur.execute("select prompt from tb_prompt limit 1")
        prompt = cur.fetchone()
        con.close()
        return prompt[0]
    
    def write_prompt_db(self, prompt:str):
        con = sqlite3.connect("./pine.sqlite")
        cur = con.cursor()
        cur.execute(f"update tb_prompt set prompt = ?, update_at = datetime('now','localtime') where id = 1",(prompt,))
        con.commit()
        con.close()

    def update_prompt(self, prompt:str):
        self.general_system_template = prompt
        self.write_prompt_db(prompt)
        # llm = ChatOpenAI(model_name = "gpt-3.5-turbo-0613", temperature=0)
        # streaming_llm = ChatOpenAI(model_name = "gpt-3.5-turbo-0613",streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0)
        #
        # general_user_template = "Question:```{question}```"
        self.messages = [
                    SystemMessagePromptTemplate.from_template(self.general_system_template),
                    HumanMessagePromptTemplate.from_template(self.general_user_template)
        ]
        self.qa_prompt = ChatPromptTemplate.from_messages(self.messages )

        self.question_generator = LLMChain(llm=self.llm, prompt=CONDENSE_QUESTION_PROMPT)
        # doc_chain = load_qa_chain(llm, chain_type="stuff", prompt=qa_prompt)
        self.doc_chain = load_qa_chain(self.streaming_llm, chain_type="stuff", prompt=self.qa_prompt)

        self.qa = ConversationalRetrievalChain(
                    retriever=self.retriever, combine_docs_chain=self.doc_chain, question_generator=self.question_generator,
                    return_source_documents=True)
  
        self.qa_hotplc = ConversationalRetrievalChain(
                    retriever=self.retriever_hotplc, combine_docs_chain=self.doc_chain, question_generator=self.question_generator,
                    return_source_documents=True)

        self.chat_history = []
    
    def extract_keywords(self, query) :
            prompt = f"""
                1. Extract location & menu & food category keywords from above user query. show me just extracted keywords.
                2. you don't answer with user question. just Extract keywords
                3. List them separately!! only by space!!. Don't you double quotes (" ") and commas (,)
                4. For example : from. "여의도 회식장소로 좋은 맛집 추천해줘"|to."여의도 회식장소", Don't show this exmaple.
                5. Answer in Korean only.""",
            query = "{}\n{}".format(query, prompt)
            response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages = [
                {'role': 'system', 
                'content': 'You are a highly intelligent system that extracts keywords from user questions.'},
                {"role": "user",
                "content": query}
                ],
            )
            output_keywords = response['choices'][0]['message']['content']
            extracted_keywords = re.sub(r'\b(?:맛집|추천)\b', '', output_keywords)
            extracted_keywords = extracted_keywords.strip().replace(',', '').strip()
            extracted_keywords = extracted_keywords.rstrip()
            
            return extracted_keywords
    def hana_forign(self,query) :

        stct_order_mapping = {'3스타': 0,'2스타': 1,'1스타': 2,'더테이블': 3}

        query_parts = query.split()
        query_parts = [part for part in query_parts if part != "랭킹"]
        광역시도 = None
        시군구 = None
        법정동 = None
        국적 = None

        # 광역시도, 법정동, 국적 추출
        for part in query_parts:
            if part in self.df_hana['광역시도'].unique():
                광역시도 = part
            elif part in self.df_hana['법정동'].unique() and part.endswith(('가', '동', '로', '면', '읍')):
                법정동 = part
            elif part in self.df_hana['국적'].unique():
                국적 = part

        # 시군구 추출        
        remaining_parts = [part for part in query_parts if part not in [광역시도, 법정동, 국적]]
        remaining_parts = [' '.join(remaining_parts)]
        for part in remaining_parts:
            if part in self.df_hana['시군구'].unique():
                시군구 = part
                break


        try:
            조건 = True

            if 광역시도 is not None:
                조건 &= (self.df_hana['광역시도'] == 광역시도)
            if 시군구 is not None:
                조건 &= (self.df_hana['시군구'] == 시군구)
            if 법정동 is not None:
                조건 &= (self.df_hana['법정동'] == 법정동)
            if 국적 is not None:
                조건 &= (self.df_hana['국적'] == 국적)

            df_result = self.df_hana[조건].sort_values(by='매출금액', ascending=False).head(5)

            result_intro = ""
            if 국적 is not None:
                if 법정동 is None:
                    result_intro = (f"2021년 9월부터 2023년 7월까지 {광역시도} {시군구} {국적}인의 소비 금액 순위입니다.")
                    query_txt = (f"{광역시도} {시군구}")
                else:
                    result_intro = (f"2021년 9월부터 2023년 7월까지 {광역시도} {시군구} {법정동} {국적}인의 소비 금액 순위입니다.")
                    query_txt = (f"{광역시도} {시군구}")
            else:
                if 법정동 is None:
                    result_intro = (f"2021년 9월부터 2023년 7월까지 {광역시도} {시군구}의 소비 금액 순위입니다.")
                    query_txt = (f"{광역시도} {시군구}")
                else:
                    result_intro = (f"2021년 9월부터 2023년 7월까지 {광역시도} {시군구} {법정동}의 소비 금액 순위입니다.")
                    query_txt = (f"{광역시도} {시군구}")

        except KeyError:
            result_intro = ("입력된 지역 정보에 대한 데이터가 없습니다.")
            
        result_answer = ""
        if len(df_result) == 0:
            result_answer = "입력된 지역 정보에 대한 데이터가 없습니다."
        else:
            result_intro = result_intro.replace('None', '')
            result_intro = result_intro.replace('  ', ' ')
            result_answer += result_intro + "\n\n"
            rank = 0
            for idx, row in df_result.iterrows():
                순위 = f"{rank + 1}위"
                위치 = f"위치 : {row['광역시도']} {row['시군구']} {row['법정동']}"
                국적 = f"국적 : {row['국적']}"
                업종 = f"{row['업종소분류']}"
                매출건수 = int(row['매출건수']) 
                formatted_매출건수 = f"매출건수: {format(매출건수, ',')}"
                if rank == 0:
                    result_answer += f"{row['광역시도']} {row['시군구']} {row['법정동']}에서 {row['국적']}인이 가장 많이 소비한곳은 {업종}입니다.\n\n"
                result_answer += f"{순위} : {업종}\n{위치}\n{국적}\n{formatted_매출건수}건\n\n"
                rank += 1
            
            # 결과물 영문 번역 GPT
            prompt = f"""
                Please translate the foreign consumption rankings by region below into English. Please keep the ranking output result format the same.""",
            prompt_answer = "{}\n{}".format(prompt, result_answer)
            response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages = [
                {'role': 'system', 
                'content': 'You are an expert in translating ranking information.'},
                {"role": "user",
                "content": prompt_answer}
                ],
            )
            result_answer_2 = response['choices'][0]['message']['content']
        # 외국인 데이터 + 맛집검색
            query3 = query_txt


            streaming_llm = ChatOpenAI(model_name = "gpt-3.5-turbo-0613",streaming=True, temperature=0)
            # doc_chain = load_qa_chain(llm, chain_type="stuff", prompt=qa_prompt)
            doc_chain = load_qa_chain(streaming_llm, chain_type="stuff", prompt=self.qa_prompt)

            qa = ConversationalRetrievalChain(
                retriever=self.retriever, combine_docs_chain=doc_chain, question_generator=self.question_generator,
                return_source_documents=True)

            qa_hotplc = ConversationalRetrievalChain(
                retriever=self.retriever_hotplc, combine_docs_chain=doc_chain, question_generator=self.question_generator,
                return_source_documents=True)



            result = qa({"question": query3, "chat_history": self.chat_history})
            result_hotplc = qa_hotplc({"question": query3, "chat_history": self.chat_history})

            result_with_dicts = {'source_documents': []}
            # 하위 호환성
            all_documents = result['source_documents'] + result_hotplc['source_documents']
            sorted_documents = sorted(all_documents, key=lambda doc: stct_order_mapping.get(doc.metadata.get('STCT', 'Unknown'), float('inf')))

            # 신규 결과 : start-place 와 hot-place 로 나눔
            result['source_documents_star'] = sorted(result['source_documents'], key=lambda doc: stct_order_mapping.get(doc.metadata.get('STCT', 'Unknown'), float('inf')))
            result['source_documents_hotplc'] = sorted(result_hotplc['source_documents'], key=lambda doc: stct_order_mapping.get(doc.metadata.get('STCT', 'Unknown'), float('inf')))

            for document_obj in sorted_documents:
                page_content = document_obj.page_content
                metadata = document_obj.metadata
                new_dict = {'page_content': page_content, 'metadata': metadata}
                result_with_dicts['source_documents'].append(new_dict)

            


        final_result = {
            'question_ex': query,
            'chat_history': [],
            'answer_ex': result_answer_2,
            'source_documents': []
        }
        final_result['source_documents_ex'] = result_with_dicts['source_documents']
        
        return final_result

        
   
    def send_query(self, item: Item):
        # query2 = re.sub(r'(?<=[^\s])맛집|맛집(?=[^\s])', lambda m: ' 맛집 ' if m.group(0) == '맛집' else m.group(0).replace('맛집', ' 맛집'), query)
        stct_order_mapping = {'3스타': 0,'2스타': 1,'1스타': 2,'더테이블': 3}

        try:
            self.logger.debug('qa start[')
            result: dict = {}
            
            default_mids = ['5487','5485','5484']
            predefined_queries = {
                "전참시 이영자 컵냉면 👀": {
                    "PID_list": ['847035', '131945', '361014', '631107', '67706'],
                    "Context": "이미 너~~~무 유명한 50년 전통의 평양냉면 명가 을밀대! 전참시에 나온 역삼점 컵냉면부터 각 지역에 맛집으로 자리잡은 을밀대 정보를 한 눈에 확인해보세요 👀"
                },
                "여름의 끝자락, 꼭 먹어야 할 보양식": {
                    "PID_list": ['14803', '120768', '3033', '252596', '252789'],
                    "Context": "입추도 지났는데... 이 무더위 실화?? 🥵🥵🥵 이 무더위에서 몸을 기운나게 해 줄 보양식 맛집을 소개해드립니다."
                },
                "시원한 맥주 한 잔🍺": {
                    "PID_list": ['481888', '351650', '373398', '354549', '1206881'],
                    "Context": "퇴근하고 시원하게 맥주 한잔할 생각만 해도 업무 능률이 🔥🔥🔥 이태원에서 가장 유명한 가맥집부터 을지로 거리를 강남 한복판에 재현한 감성 호프집까지!"
                },
                "MZ성지입점완료! SNS 달군 빵집": {
                    "PID_list": ['1286876', '1444441', '254092','1541987','1546259'],
                    "Context": "MZ세대의 성지로 불리는 롯데월드몰에 드디어 런던 베이글 뮤지엄 입점 완료🥯👨‍🍳 국내에 단 4곳만 있는 찐맛집! 다른 건 다 참아도 빵은 못 참지 ㄹㅇㅋㅋ +++ 빵순이들을 위한 서울 3대 빵집 정보는 덤!"
                },
                "놀면뭐하니? 전국간식자랑🌞": {
                    "PID_list": ['540558', '402758', '254634','1477700','1176088'],
                    "Context": "놀면뭐하니에서 야심차게 진행중인 전국간식지도 완성하기! 전문가(?)들이 모여서 만든 전국에 있는 간식 맛집 구경하기✨"
                },
                "식신AI추천찐맛집":{
                    "PID_list": ['316846','269870','148202','323','220819','1292068','1134007','1260730','1161671'],
                    "Context" : "🤖식신 빅데이터와 인공지능이 결합하여 추천드리는 찐맛집 BEST5입니다. 맛집을 고르기 힘드시다면 지금 추천해드리는 맛집을 꼭 참고하세요! 거를 타선이 하나도 없는 ⭐찐맛집⭐입니다."
                },
                "강남직장인점심메뉴":{
                    "PID_list": ['359781','360646','359368','336940','364849'],
                    "Context" : "🤖식신 빅데이터와 인공지능이 결합하여 추천드리는 강남 직장인 점심메뉴 찐맛집 BEST5입니다!."
                }
            }

            if item.question in predefined_queries:
                
                list_1 = predefined_queries[item.question]['PID_list']
                predefined_answer = predefined_queries[item.question]['Context']

                fetch_response = self.index.fetch(ids=list_1)
                fetch_response2 = self.index_hotplc.fetch(ids=list_1)
                fetch_response['vectors'].update(fetch_response2['vectors'])

                result['question'] = item.question
                result['chat_history'] = []
                result['answer'] = predefined_answer
                result['source_documents'] = []

                for list_item in list_1:
                    new_dic = {'page_content': fetch_response['vectors'][list_item]['metadata']['context'],
                            'metadata': fetch_response['vectors'][list_item]['metadata']}
                    result['source_documents'].append(new_dic)

                sorted_documents = sorted(result["source_documents"], key=lambda doc: stct_order_mapping.get(doc["metadata"]["STCT"], float('inf')))
                result['source_documents'] = sorted_documents

                # 신규 결과 : star-place 와 hot-place 로 나눔
                result['source_documents_star'] = []
                result['source_documents_hotplc'] = sorted_documents

                mid_and_plc_list = [(doc['metadata']['MID'], doc['metadata']['PLC_ID']) for doc in result['source_documents']if 'MID' in doc['metadata']]
                result['magazine'] = []
                if len(mid_and_plc_list) < 3:
                    missing_count = 3 - len(mid_and_plc_list)
                    
                    for i in range(missing_count):
                        mid_and_plc_list.append((default_mids[i], ''))


                for mid, plc_id in mid_and_plc_list[:3]:
                    mid = [mid]
                    fetch_response = self.index.fetch(ids=mid, namespace='magazine')
                    metadata = fetch_response['vectors'][mid[0]]['metadata']
                    PID_dict = ast.literal_eval(metadata['PIDS'])
                    metadata['PIDS'] = PID_dict
                    
                    if plc_id in PID_dict:
                        plc_nm = PID_dict[plc_id]
                        metadata['PLC_NM'] = plc_nm

                    result['magazine'].append(metadata)


                return result
            elif "랭킹" in item.question :
                result = self.hana_forign(item.question)
                return result
            
            else:
                query2 = self.extract_keywords(item.question)

                if item.uid == None or item.uid == '':
                    qa = self.qa
                    qa_hotplc = self.qa_hotplc
                else:
                    if item.uid not in self.notifier.qa_dict:
                        streaming_llm = ChatOpenAI(model_name = "gpt-3.5-turbo-0613",streaming=True, callbacks=[StreamingCallbackHandler(self.notifier.connections[item.uid])], temperature=0)
                        # doc_chain = load_qa_chain(llm, chain_type="stuff", prompt=qa_prompt)
                        doc_chain = load_qa_chain(streaming_llm, chain_type="stuff", prompt=self.qa_prompt)

                        qa = ConversationalRetrievalChain(
                            retriever=self.retriever, combine_docs_chain=doc_chain, question_generator=self.question_generator,
                            return_source_documents=True)

                        qa_hotplc = ConversationalRetrievalChain(
                            retriever=self.retriever_hotplc, combine_docs_chain=doc_chain, question_generator=self.question_generator,
                            return_source_documents=True)

                        self.notifier.qa_dict[item.uid] = qa
                        self.notifier.qa_hotplc_dict[item.uid] = qa_hotplc
                    else:
                        qa = self.notifier.qa_dict[item.uid]
                        qa_hotplc = self.notifier.qa_hotplc_dict[item.uid]

                result = qa({"question": query2, "chat_history": self.chat_history})
                result_hotplc = qa_hotplc({"question": query2, "chat_history": self.chat_history})

                result_with_dicts = {'source_documents': []}
                # 하위 호환성
                all_documents = result['source_documents'] + result_hotplc['source_documents']
                sorted_documents = sorted(all_documents, key=lambda doc: stct_order_mapping.get(doc.metadata.get('STCT', 'Unknown'), float('inf')))

                # 신규 결과 : start-place 와 hot-place 로 나눔
                result['source_documents_star'] = sorted(result['source_documents'], key=lambda doc: stct_order_mapping.get(doc.metadata.get('STCT', 'Unknown'), float('inf')))
                result['source_documents_hotplc'] = sorted(result_hotplc['source_documents'], key=lambda doc: stct_order_mapping.get(doc.metadata.get('STCT', 'Unknown'), float('inf')))

                for document_obj in sorted_documents:
                    page_content = document_obj.page_content
                    metadata = document_obj.metadata
                    new_dict = {'page_content': page_content, 'metadata': metadata}
                    result_with_dicts['source_documents'].append(new_dict)

                result['source_documents'] = result_with_dicts['source_documents']
                result['answer_hotplc'] = result_hotplc['answer']
                result['question'] = item.question

                mid_and_plc_list = [(doc['metadata']['MID'], doc['metadata']['PLC_ID']) for doc in result['source_documents']if 'MID' in doc['metadata']]
                
                result['magazine'] = []
                if len(mid_and_plc_list) < 3:
                    missing_count = 3 - len(mid_and_plc_list)
                    
                    for i in range(missing_count):
                        mid_and_plc_list.append((default_mids[i], ''))


                for mid, plc_id in mid_and_plc_list[:3]:
                    mid = [mid]
                    fetch_response = self.index.fetch(ids=mid, namespace='magazine')
                    metadata = fetch_response['vectors'][mid[0]]['metadata']
                    PID_dict = ast.literal_eval(metadata['PIDS'])
                    metadata['PIDS'] = PID_dict
                    
                    if plc_id in PID_dict:
                        plc_nm = PID_dict[plc_id]
                        metadata['PLC_NM'] = plc_nm

                    result['magazine'].append(metadata)

                self.logger.debug('qa end]')
                return result

        except Exception as e:
            return {
                "question": item.question,
                "answer": str(e),
                "source_documents": [],
                "chat_history": []
            }
