from langchain_core.output_parsers import StrOutputParser

class AnswerOnlyParser(StrOutputParser):
    def parse(self, text: str) -> str:
        # Assuming the answer always follows "Answer:"
        start = text.find("Answer:") + len("Answer:")
        end = text.find("History message:", start)
        if end == -1:
            end = len(text)
        answer = text[start:end].strip()
        return answer
    
    
def message_with_histories(messages):
    histories = [f"{message['role'].capitalize()}: {message['content']}" for message in messages]
    histories = "\n".join(histories)
    return histories

def format_docs(docs):
    return "\n".join(doc.page_content.replace("\n", "").strip() for doc in docs)


def format_documents(documents):
    file_name = documents[0].metadata["source"].split("/")[-1]
    page_number = documents[-1].metadata["page"]
    text_document = f"Provided file name is {file_name}, its has {page_number} pages.\n"
    for doc in documents:
        text_document += f"page {doc.metadata['page']}: {doc.page_content}\n"
    return text_document