from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from graph_retriever import CustomGraphRAGRetriever
from langchain_ollama import ChatOllama

class CodeReviewState(TypedDict):
    pr_diff: str                 # Input: Nội dung Git Diff
    changed_files: List[str]     # Danh sách file/hàm thay đổi
    context_data: List[str]      # Dữ liệu lấy từ GraphRAG
    final_review: str            # Output: Kết quả review
class GraphRAGBot:
    def __init__(self):
        self.app = None
        self.retriever = None
        self.llm = None

    def initialize(self):
        self.retriever = CustomGraphRAGRetriever()
        self.llm = ChatOllama(
            model="qwen2.5-coder:7b",
            temperature=0,     
        )

        workflow = StateGraph(CodeReviewState)
        workflow.add_node("parse", self.parse_diff_node)
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("review", self.review_node)

        workflow.set_entry_point("parse")
        workflow.add_edge("parse", "retrieve")
        workflow.add_edge("retrieve", "review")
        workflow.add_edge("review", END)

        self.app = workflow.compile()

    def invoke(self, inputs):
        if not self.app:
            raise Exception("Bot chưa được initialize!")
        return self.app.invoke(inputs)

    def close(self):
        if self.retriever:
            self.retriever.close()
    
    def parse_diff_node(self, state: CodeReviewState):
        print("--- STEP 1: PARSING DIFF ---")
        diff = state["pr_diff"]
        # co thể dùng thư viện `unidiff`
        queries = []
        lines = diff.split('\n')
        for line in lines:
            if line.startswith('def ') or line.startswith('class '):
                queries.append(line.strip())
            if "config" in line or "yaml" in line:
                queries.append("configuration settings")

        # Nếu không parse được gì cụ thể, dùng cả đoạn diff làm query tìm kiếm
        if not queries:
            queries = [diff[:200]] 
            
        return {"changed_files": queries}

    def retrieve_node(self,state: CodeReviewState):
        print("--- STEP 2: RETRIEVING GRAPH CONTEXT ---")
        queries = state["changed_files"]
        collected_context = []
        
        for query in queries:
            results = self.retriever.search(query_text=query, top_k=5)
            collected_context.extend(results)
            
        return {"context_data": collected_context}

    def review_node(self,state: CodeReviewState):
        print("--- STEP 3: GENERATING REVIEW ---")
        system_prompt = """Bạn là Senior Code Reviewer & Security Auditor.
        
        Nhiệm vụ: Review đoạn code diff dựa trên NGỮ CẢNH HỆ THỐNG được cung cấp.
        
        NGỮ CẢNH HỆ THỐNG (Từ Knowledge Graph):
        {graph_context}
        
        PULL REQUEST DIFF:
        {pr_diff}
        
        HƯỚNG DẪN REVIEW:
        1. **Endpoint & Security**: Nếu thay đổi liên quan đến Endpoint, hãy kiểm tra xem nó có gọi hàm xác thực (Auth) nào trong ngữ cảnh không?
        2. **Configuration Impact**: Nếu thay đổi liên quan đến Configuration, hãy cảnh báo tất cả các hàm (Methods) đang sử dụng config đó.
        3. **Logic Flow**: Kiểm tra các hàm gọi (Callers) để đảm bảo thay đổi không phá vỡ logic cũ.
        
        Hãy trả về định dạng Markdown, chia rõ các mục: [Tóm tắt], [Phân tích tác động], [Cảnh báo bảo mật], [Đề xuất].
        """
        
        prompt = ChatPromptTemplate.from_template(system_prompt)
        chain = prompt | self.llm
        
        context_str = "\n".join(state["context_data"]) if state["context_data"] else "Không tìm thấy ngữ cảnh trong Graph."
        
        response = chain.invoke({
            "graph_context": context_str,
            "pr_diff": state["pr_diff"]
        })
        
        return {"final_review": response.content}

bot_instance = GraphRAGBot()



