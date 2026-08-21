## csv 파일 로드 후 document list로 변환
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader
import os

polictech_docs = [] # 폴리텍대학 학과 document 리스트
jobtraining_docs = [] # 직업훈련 기관 document 리스트

polictech_loader = CSVLoader(file_path="./data/학교법인한국폴리텍_교육훈련계획_20250916.csv",
                             content_columns=["캠퍼스", "캠퍼스위치", "기간(개월)", "전공계열구분", "전공직종", "전공설명", "전공성향", "주간야간구분"]) # 폴리텍 학과목록 CSV

jobtraining_loader = CSVLoader(file_path="./data/한국고용정보원_직업훈련 기관 목록_20250910.csv",
                               content_columns=["훈련기관명", "훈련기관위치", "훈련기관계열구분"]) # 직업훈련 기관목록 CSV

polictech_docs.extend(polictech_loader.load())
jobtraining_docs.extend(jobtraining_loader.load())

## embedding 모델 생성
embeddings = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask",
    encode_kwargs={"normalize_embeddings":True}
)

## 벡터 DB 구축 (싱글톤 패턴 적용)
# 벡터 DB 최초 생성
def create_polictech_db():
    vectorstore = FAISS.from_documents(polictech_docs, embeddings)
    vectorstore.save_local("PolitechDB")
    return vectorstore
def create_jobtraining_db():
    vectorstore = FAISS.from_documents(jobtraining_docs, embeddings)
    vectorstore.save_local("JobTrainingDB")
    return vectorstore

# 벡터 DB 로드
def load_polictech_db():
    vectorstore = FAISS.load_local("PolitechDB", embeddings, allow_dangerous_deserialization=True)
    return vectorstore
def load_jobtraining_db():
    vectorstore = FAISS.load_local("JobTrainingDB", embeddings, allow_dangerous_deserialization=True)
    return vectorstore

# 벡터 DB 가져오기
def get_polictech_db():
    if os.path.exists("PolitechDB"):
        return load_polictech_db()
    else:
        return create_polictech_db()
def get_jobtraining_db():
    if os.path.exists("JobTrainingDB"):
        return load_jobtraining_db()
    else:
        return create_jobtraining_db()