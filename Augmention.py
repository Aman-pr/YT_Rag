from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

def Augmention(context, Question):
    llm = ChatGroq(model="llama-3.3-70b-versatile")
    parser = StrOutputParser()

    template = PromptTemplate(
        template="""
You are a helpful assistant.
Answer ONLY from the provided Transcript Context.
If the Context is insufficient, say "I don't know".

Context:
{Context}

Question:
{Question}
""",
        input_variables=["Context", "Question"]
    )

    chain = template | llm | parser
    result = chain.invoke({
        "Context": context,
        "Question": Question
    })

    return result



if __name__ == "__main__":
    story="""Every night, the city’s oldest bridge whispered equations to Mira. Steel hummed, rivets clicked, traffic solved for x. She was an engineering 
    student by day, exhausted, broke, stubbornly curious. One stormy 
    evening, lightning struck, and the bridge finally spoke plainly: it was tired of holding everyone together.
      Mira sketched a solution in rain, recalculating stress, sharing the load. At dawn, 
    the bridge felt lighter. Commuters crossed unaware, but the structure sang softly. Mira smiled, knowing some problems don’t need applause
    , only balance, patience, and the courage to listen. That night, she slept well, dreaming of systems healed by quiet care.

    """
    question="""What problem did Mira solve for the bridge, 
    and how did her engineering mindset help fix it?"""
    Augmention(story,question)
    print("checkpoitn")