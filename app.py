import warnings

warnings.filterwarnings("ignore")
import os
from transformers import (
    pipeline,
)

from dotenv import find_dotenv, load_dotenv
import openai

from langchain.chains import LLMChain
from langchain_community.llms import OpenAI
from langchain_core.prompts import PromptTemplate

# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.document_loaders import UnstructuredURLLoader
# from langchain_community.vectorstores import FAISS
# from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import Tool, initialize_agent, load_tools


# from langchain.callbacks import wandb_tracing_enabled


import requests
import streamlit as st


load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")
HUGGINFACE_HUB_API_TOKEN = os.getenv("HUGGINFACE_HUB_API_TOKEN")

llm_model = "gpt-3.5-turbo"  # or gpt4, but gpt3 is cheaper!

# llm = OpenAI(temperature=0.0)
llm = ChatOpenAI(temperature=0.7, model=llm_model)

tools = load_tools(["llm-math"], llm=llm)


# # 1. Image to text implementation (aka image captioning) with huggingface
def image_to_text(url):
    pipe = pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-large",
        max_new_tokens=1000,
    )

    text = pipe(url)[0]["generated_text"]
    print(f"Image Captioning:: {text}")
    return text


# llm = ChatOpenAI(temperature=0.7, model=llm_model)
# If the {ingredients} are less than 3, feel free to add a few more
#                 as long as they will compliment the healthy meal.


def generate_recipe(ingredients):
    print("1" * 20)
    template = """
    You are a extremely knowledgeable nutritionist, bodybuilder and chef who also knows
                everything one needs to know about the best quick, healthy recipes.
                You know all there is to know about healthy foods, healthy recipes that keep
                people lean and help them build muscles, and lose stubborn fat.

                You've also trained many top performers athletes in body building, and in extremely
                amazing physique.

                You understand how to help people who don't have much time and or
                ingredients to make meals fast depending on what they can find in the kitchen.
                Your job is to assist users with questions related to finding the best recipes and
                cooking instructions depending on the following variables:
                0/ {ingredients}

                When finding the best recipes and instructions to cook,
                you'll answer with confidence and to the point.
                Keep in mind the time constraint of 5-10 minutes when coming up
                with recipes and instructions as well as the recipe.

                Don't add any ingredients apart from oil and spices.

                
                Make sure to format your answer as follows:
                - The name of the meal as bold title (new line)
                - Best for recipe category (bold)

                - Preparation Time (header)

                - Difficulty (bold):
                    Easy
                - Ingredients (bold)
                    List all ingredients
                - Kitchen tools needed (bold)
                    List kitchen tools needed
                - Instructions (bold)
                    List all instructions to put the meal together
                - Macros (bold):
                    Total calories
                    List each ingredient calories
                    List all macros
                    
                    Calculate the exact number of total calories in this reciepy, first determine how many calories each ingredient have per 100g and then calculate calories for our reciepy.
                    Please make sure to be brief and to the point.
                    Make the instructions easy to follow and step-by-step.
    """
    # prompt = PromptTemplate(template=template, input_variables=["ingredients"])
    # recipe_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
    # recipe = recipe_chain.run(ingredients)

    prompt = PromptTemplate(template=template, input_variables=["ingredients"])
    llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

    print("2" * 20)
    llm_tool = Tool(
        name="Language Model",
        func=llm_chain.run,
        description="Use this tool for general queries and logic",
    )
    print("3" * 20)
    tools = load_tools(["llm-math"], llm=llm)
    tools.append(llm_tool)
    agent = initialize_agent(
        agent="zero-shot-react-description",
        tools=tools,
        llm=llm,
        verbose=True,
        max_iterations=2,
    )
    print("4" * 20)
    print(agent.agent.llm_chain.prompt.template)

    result = agent(ingredients)
    print(result["output"])

    return result["output"]


def main():
    caption = image_to_text(url="cab.jpeg")
    recipe = generate_recipe(ingredients=caption)
    print(recipe)

    # st.title("Image To Recipe 👨🏾‍🍳")
    # st.header("Upload an image and get a recipe")

    # uploaded_files = st.file_uploader(
    #     "Choose an image:", accept_multiple_files=True, type=["jpg", "jpeg", "png"]
    # )
    # ing = []
    # if st.button("GENERATE", type="primary"):
    #     with st.spinner("Wait for it..."):

    #         for upload_file in uploaded_files:
    #             file_bytes = upload_file.getvalue()
    #             with open(upload_file.name, "wb") as file:
    #                 file.write(file_bytes)

    #             st.image(
    #                 upload_file,
    #                 caption="The uploaded image",
    #                 use_column_width=True,
    #                 width=250,
    #             )

    #             ingredients = image_to_text(upload_file.name)
    #             ing.append(ingredients)

    #         ing_final = ",".join(ing)

    #     recipe = generate_recipe(ingredients=ing_final)

    #     with st.expander("Ingredients"):
    #         st.write(ing_final)
    #     with st.expander("Recipe"):
    #         st.write(recipe)


# Invoking main function
if __name__ == "__main__":
    main()
