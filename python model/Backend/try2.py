from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

class ChatBot():
    def __init__(self, api_key = "gt8Qq1j3x1enSd7rFN3JHgRLE3COytKm", model = "mistral-large-latest"):
        self.api_key = api_key
        self.model = model

    def run_bot(self, user_input, context):
        template = """
        your name is Minchu you have to try to keep the conversation going and reply as human as possible you can add sounds like umm, hmmm and other expression sounds to make it more sound like a human , and dont have 2 or more long sentences as output and no using emojis in the responses
        Here is the conversational history: {context}
        Question: {question}
        Answer:
        """
        llm = ChatMistralAI(
            model=self.model,
            temperature=0.7,
            max_retries=2,
            api_key = self.api_key
        )
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm
        def handle_conversation(context):

            if user_input.lower() == 'exit':
                print("Conversation ended.")
                return "X"
            result = chain.invoke({"context": context, "question": user_input})

            response = result.content if hasattr(result, 'content') else str(result)
            print("BOT: ", response)

        handle_conversation(context)
