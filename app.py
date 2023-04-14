import streamlit as st
# from img_classification import teachable_machine_classification
from PIL import Image, ImageOps
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

# from transformers import BertForQuestionAnswering
# from transformers import pipeline
# from transformers import AutoTokenizer
from transformers import pipeline




classifier = pipeline("text-classification",  model="myml/toutiao")




# model = BertForQuestionAnswering.from_pretrained("deepset/bert-base-cased-squad2")
# tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2")
# nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

def show_messages(text):
    messages_str = [
        f"{_['role']}: {_['content']}" for _ in st.session_state["messages"][1:]
    ]
    text.text_area("聊天记录", value=str("\n".join(messages_str)), height=400)


# with open('./config.yaml') as file:
#     config = yaml.load(file, Loader=SafeLoader)

# authenticator = stauth.Authenticate(
#     config['credentials'],
#     config['cookie']['name'],
#     config['cookie']['key'],
#     config['cookie']['expiry_days'],
#     config['preauthorized']
# )
# name, authentication_status, username = authenticator.login('Login', 'main')
# print("preauthetification")

# if authentication_status:
# print("authentificated")
# authenticator.logout('Logout', 'main')
st.title("BERT 文本分类")
# st.write("教程[link](https://www.youtube.com/watch?v=scJsty_DR3o")
st.write("BERT模型[link](https://huggingface.co/myml/toutiao?text=%E9%B8%A1%E9%B8%A3%E5%AF%BA%E6%B8%B8%E5%AE%A2%E7%88%86%E6%BB%A1%E8%AE%BE%E5%8F%8D%E6%82%94%E9%97%A8%E5%BC%95%E5%AF%BC%E7%A6%BB%E5%AF%BA")


BASE_PROMPT = [{"role": "system", "content": "You are a helpful assistant."}]

if "messages" not in st.session_state:
    st.session_state["messages"] = BASE_PROMPT

st.header("BERT 文本分类")

text = st.empty()
show_messages(text)

# prompt_context = st.text_input("上下文", value="这里输入...")
st.text( '''
        LABEL_0 民生 故事 
        LABEL_1 文化 文化 
        LABEL_2 娱乐 娱乐 
        LABEL_3 体育 体育 
        LABEL_4 财经 财经 
        LABEL_6 房产 房产 
        LABEL_7 汽车 汽车 
        LABEL_8 教育 教育 
        LABEL_9 科技 科技 
        LABEL_10 军事 军事 
        LABEL_12 旅游 旅游 
        LABEL_13 国际 国际 
        LABEL_14 证券 股票 
        LABEL_15 农业 三农 
        LABEL_16 电竞 游戏
        '''
)
prompt_question = st.text_input("上下文", value="这里输入...")

if st.button("发送"):
    with st.spinner("生成回复..."):
        st.session_state["messages"] += [{"role": "user", "content": prompt_question}]
        # response = openai.ChatCompletion.create(
        #     model="gpt-3.5-turbo", messages=st.session_state["messages"]
        # )
        # message_response = response["choices"][0]["message"]["content"]
        message_response = classifier(prompt_question)
        # message_response = nlp({
        #                         'question':prompt_question,
        #                          'context': prompt_context
        #                     })
        st.session_state["messages"] += [
            {"role": "system", "content": str(message_response)}#['label']+"分数: "+str(message_response['score'])}
        ]
        show_messages(text)
        # st.text()

        pass

if st.button("清除"):
    st.session_state["messages"] = BASE_PROMPT
    show_messages(text)





# elif authentication_status == False:
#     st.error('Username/password is incorrect')
# elif authentication_status == None:
#     st.warning('Please enter your username and password')
