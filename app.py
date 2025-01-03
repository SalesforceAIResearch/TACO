import streamlit as st
import pickle
import numpy as np
import os
import re
from streamlit_image_select import image_select

st.set_page_config(layout="wide")

@st.cache_data
def load_messages(model, index):
    root_path = "examples"
    messages_file = os.path.join(root_path, model, f"{index}/all_messages.pkl")
    with open(messages_file, 'rb') as file:
        messages = pickle.load(file)
    return messages

def main():
    st.title("🌮 TACO 🌮")
    st.subheader("Learning Multi-modal Action Models with Synthetic Chains-of-Thought-and-Action")
    st.markdown("Select a TACO and see its responses on different examples below!")
    
    model = st.selectbox(
        "Model",
        ("TACO-Qwen2-SigLIP", "TACO-Llama3-SigLIP"),
    )
    all_indices = [7, 10, 11, 12, 13, 14, 15, 20, 21, 28, 33, 36, 38, 41, 42, 44, 46, 47, 51, 53, 54, 60, 61, 65, 69, 72, 74, 75, 76, 77, 81, 86, 89, 90, 91, 92, 94, 95, 97, 99, 100, 101, 102, 110, 111, 112, 113, 114, 115, 116, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 131, 133, 134, 135, 136, 137, 138, 140, 143, 147, 149, 151, 152, 156, 157, 166, 179, 181, 182, 183, 199, 200, 203, 210, 211, 213, 216, 218]
    
    shuffle_btn = st.button("Shuffle Examples", type="primary")
    n = 7
    if "random_indices" not in st.session_state:
        st.session_state.random_indices = np.random.choice(all_indices, n, replace=False)
    if shuffle_btn:
        st.session_state.random_indices = np.random.choice(all_indices, n, replace=False)

    images = [f"examples/MMVet/{index}.jpg" for index in st.session_state.random_indices]
    img = image_select("Examples", images)
    # st.write(img)
    index = st.session_state.random_indices[images.index(img)]
    all_msgs = load_messages(model, index)
    # st.write(all_msgs)
    msgs = all_msgs["user_agent"]
    
    st.write(f"You are viewing {model}'s reponse on example {index} of the MMvet dataset.")
    for i, msg in enumerate(msgs):
        if i == 0:
            for j, content in enumerate(msg["content"]):
                if j == 0: continue  # skip the user request prefix
                with st.chat_message(msg["role"]):
                    for k, v in content.items():
                        if j == len(msg["content"]) - 1: 
                            v = v.replace("Now please generate your response:", "")
                        if k == "type": 
                            continue
                        elif k == "image_url": 
                            image = v["url"]
                            image_arr = np.asarray(image)
                            st.image(image_arr, width=400)
                        else: # text
                            st.write(v)
        else:
            with st.chat_message(msg["role"], avatar="🌐" if msg["role"] == "user" else None):
                if len(msg["content"]) > 1:
                    vals = []
                    # concatenate all the values into a string
                    pil_image = None
                    for content in msg["content"]:
                        for k, v in content.items():
                            if k == "type": continue
                            elif k == "image_url":
                                pil_image = v["url"]
                                v = str(v["url"]) # turn PIL object into a string
                            vals.append(v)
                    v_str = "".join(vals)
                    try:
                        start = v_str.find("OBSERVATION:") + len("OBSERVATION:")
                        end = v_str.find("The OBSERVATION can be incomplete or incorrect")
                        v_dict = eval(v_str[start:end]) # extract the observation dict
                        for k, v in v_dict.items():
                            if k == "image" and pil_image:
                                image_arr = np.asarray(pil_image) # display the PIL object cached earlier
                                st.image(image_arr, width=400)
                            else:
                                st.write(f"{k.upper()}: {v}")
                    except Exception as e:
                        print(e)
                        st.write(v_str)
                else:
                    content = msg["content"][0]
                    for k, v in content.items():
                        if k == "type": 
                            continue
                        elif k == "image_url": 
                            image = v["url"]
                            image_arr = np.asarray(image)
                            st.image(image_arr, width=400)
                        else: # text
                            if msg["role"] == "assistant": # cota 
                                v_dict = eval(v)
                                st.write("THOUGHT:", v_dict["thought"])
                                if len(v_dict["actions"]) > 0:
                                    action = v_dict["actions"][0] 
                                    arg_str = ",".join([f"{k}={v}" for k, v in action["arguments"].items()])
                                    st.write("ACTION:", f'{action["name"]}({arg_str})')
                                else:
                                    st.write("ACTION: None")
                            else:
                                try:
                                    start = v.find("OBSERVATION:") + len("OBSERVATION:")
                                    end = v.find("The OBSERVATION can be incomplete or incorrect")
                                    v_dict = eval(v[start:end]) # extract the observation dict
                                    for k, v in v_dict.items():
                                        st.write(f"{k.upper()}: {v}")
                                except Exception as e:
                                    print(e)
                                    st.write(v) 

    
if __name__ == "__main__":
    main()