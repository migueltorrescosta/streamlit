import streamlit as st

st.set_page_config(
    page_title="Quantum Home base",
    page_icon="⚛️",
    layout="wide"
)

st.header("Miguel's dashboard", divider="blue")

quicklinks = {
    "🌊️ Collapsed Wave": "https://collapsedwave.com",
}

columns = st.columns(len(quicklinks))

for i, text in enumerate(quicklinks.keys()):
    url = quicklinks[text]
    with columns[i]:
        # st.markdown("\n".join([f"- [{text}]({url})" for text, url in quicklinks.items()]))
        st.markdown(f"[{text}]({url})")


# st.markdown(
#     """
#     Streamlit is an open-source app framework built specifically for
#     Machine Learning and Data Science projects.
#     **👈 Select a demo from the sidebar** to see some examples
#     of what Streamlit can do!
#     ### Want to learn more?
#     - Check out [streamlit.io](https://streamlit.io)
#     - Jump into our [documentation](https://docs.streamlit.io)
#     - Ask a question in our [community
#         forums](https://discuss.streamlit.io)
#     ### See more complex demos
#     - Use a neural net to [analyze the Udacity Self-driving Car Image
#         Dataset](https://github.com/streamlit/demo-self-driving)
#     - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
# """
# )

