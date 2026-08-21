from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableBranch
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from dotenv import load_dotenv
from pydantic import BaseModel
from vector import get_jobtraining_db
from vector import get_polictech_db

load_dotenv()

## LLM 생성
llm = ChatOpenAI(model="gpt-4o-mini", # 내용 분석 및 요약용 LLM
                 max_tokens=350,
                 temperature=0)

answer_llm = ChatOpenAI(model="gpt-4o-mini", # 사용자 답변용 LLM
                 max_tokens=400,
                 temperature=0.4,
                 frequency_penalty=0.5)

## 5. RETRIEVAL LAYER
def retrieval_docs(filter_and_query):
    # 메타 데이터로 선 필터링
    vectorstore_filter = { k:v for k, v in filter_and_query["filter"].items() if k == "구분"}
    print(vectorstore_filter)
    
    # 검색 수행
    if "캠퍼스" in vectorstore_filter["구분"]:
        vectorstore = get_polictech_db()
        result = vectorstore.similarity_search(
            query=filter_and_query["query"],
            k=3
        )
        return result
    vectorstore = get_jobtraining_db()
    result = vectorstore.similarity_search(
        query=filter_and_query["query"],
        k=3
    )
    return result

def create_context(docs):
    data = ""
    for doc in docs:
        data = data + doc.page_content + "\n\n"
    return data

## 6. AUGMENTED LAYER
def agument_query(filter_and_query):
    context = create_context(retrieval_docs(filter_and_query))
    return {"query":filter_and_query["query"], "context":context, "base_query":filter_and_query["query"]}

# 7. GENERATION LAYER
answer_prompt = ChatPromptTemplate({
    ("system", """
     너는 직업훈련 기관 및 폴리텍대학 학과 추천 전문가야
     사용자의 질의와 참고 자료를 바탕으로 작성할거니 절대 다른건 참고하지마
     출력하기 전에 간단한 코멘트 1~2줄 작성하고 출력 형태에 따라 작성해
     캠퍼스명을 입력할 땐 "xxx 캠퍼스" 형태로 출력해
     
     [출력 형태]
     1. 전공직종 or 훈련기관명 (직종이 겹치는 경우 하나만 출력)
     - 세부 내용 (참고 자료에서 받은 내용이 모두 포함되게 출력)
     2. 전공직종 or 훈련기관명
     - 세부 내용
     3. 전공직종 or 훈련기관명
     - 세부 내용
     
     마지막으로 사용자에게 다른 분야나 전공의 폴리텍대학이나 직업훈련 기관에 관심있는지 물어봐 
          
     [참고 자료]
     {context}
     """),
    ("human","{query}")
})

answer_chain = answer_prompt | answer_llm | StrOutputParser()

## 4. REWRITER LAYER
class Rewriter(BaseModel):
    filter: dict
    query: str

rewriter_parser = JsonOutputParser(pydantic_object=Rewriter)

rewriter_prompt = ChatPromptTemplate.from_messages({
    ("system", """
    너는 직업훈련 기관 및 폴리텍대학 학과 추천 전문가야
    사용자의 질문을 분석해서 상세 조건 필터(filter)와 검색용 쿼리(str)를 만들어야 해
    
    [상세 조건 필터 작성 규칙]
    1. 구분 : 사용자 질문이 폴리텍대학 입학, 전공 및 학과 추천, 폴리텍대학을 통한 취업 등의
    폴리텍 대학과 연관된 내용이라면 "캠퍼스" 작성, 직업훈련과 관련된 내용이라면 "훈련기관" 작성
    
    [검색용 쿼리 작성 규칙]
    1. 지역명 추출 후 "서울 구로구", "경기 수원시"와 같은 형태로 작성
    1. 계열 : 전공이나 계열, 원하는 직무를 분석 후 "디자인계열", "IT계열"과 같은 "XX계열" 형식으로 무조건 작성
    2. 성향 : 사용자의 성향을 분석해 3개의 카테고리로 분석. "꼼꼼함/전문성/논리적 사고" 이런 형태로 작성
    사용자가 질문을 작성한 의도 파악하고 위 조건에 맞춰 아래와 같은 한 문장으로 작성
    3. 기간(개월): 기간을 개월로 치환하여 숫자로만 표기. 없으면 "x"
    4. 주간야간구분 : 시간 정보를 분석해 "주간", "야간" 중 선택. 정보가 없으면 "x"으로 작성
    예시: "창의성 및 전문성을 갖추는 디자인계열의 경기 수원시에 있거나 가까운 폴리텍대학 캠퍼스를 찾아줘",
    "섬세함을 요하는 IT계열의 서울 구로구에 있거나 가까운 직업훈련 기관을 찾아줘"
    
    {format_instructions}
     """),
    ("human", "{query}")
}).partial(format_instructions=rewriter_parser.get_format_instructions)

rewriter_chain = rewriter_prompt | llm | rewriter_parser

## 2. THEME_FILTER LAYER
class Theme(BaseModel):
    flag: str
    query: str

theme_parser = JsonOutputParser(pydantic_object=Theme)

theme_prompt = ChatPromptTemplate.from_messages({
    ("system", """
     아래의 문장이 직업훈련이나 폴리텍대학과 관련된 내용인지 판단하고
     결과를 yes 혹은 no로 대답하고 사용자가 입력한 쿼리문도 그대로 출력해
     {format_instructions}
     """),
    ("human","{query}")
}).partial(format_instructions=theme_parser.get_format_instructions)

theme_chain = theme_prompt | llm | theme_parser

## 1. PROFANITY_FILTER LAYER
with open("./data/bad_word.txt", "r", encoding="utf-8") as f:
        data = f.read()
        bad_words = data.split("\n")
        
# profanity filtering
def profanity_filter(input):
    for bad_word in bad_words:
        if bad_word in input["query"]:
            return {"flag":"true", "query":input["query"]}
    return {"flag":"false", "query":input["query"]}

def themeComment(x):
    comment = "안녕하세요!😃 MileVision AI입니다. 직업훈련이나 폴리텍대학을 추천받고 싶으신가요?\n'나에게 맞는 직업훈련을 추천해줘', '인천에서 다닐 수 있는 폴리텍대학이 있을까?'와 같이 질문해보세요!\n성향이나 관심사, 지역 등을 작성해주시면 좀 더 나에게 맞는 곳을 추천받을 수 있어요."
    return comment

def profanityComment(x):
    comment = "입력하신 부분에 비속어 또는 욕설이 포함되어 있습니다.\n비속어나 욕설을 제외하고 다시 작성해주세요."
    return comment

theme_branch = RunnableBranch(
    (lambda x: x["flag"]=="no", RunnableLambda(themeComment)),
    rewriter_chain
    | RunnableLambda(agument_query)
    | answer_chain
)

# profanity branch
profanity_branch = RunnableBranch(
    (lambda x: x["flag"]=="true", RunnableLambda(profanityComment)),
    theme_chain
    | theme_branch
)

start_chain = (
    profanity_filter
    | profanity_branch
)

def get_chat_response(user_query):
    return start_chain.invoke({"query":user_query})