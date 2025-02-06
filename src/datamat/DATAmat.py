import sys
from utils import setup_qa_chain

qa_chain = setup_qa_chain()

while True:
    user_input = input(f"Input Prompt:")
    if user_input == "exit":
        sys.exit()
    result = qa_chain({"query": user_input})
    print(result["result"])