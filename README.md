Problem:

The agricultural sector faces several challenges, including a growing need for up-to-date, accessible information on farming practices, crop management, pest control, and sustainable techniques. Farmers, agronomists, and agricultural enthusiasts often rely on a variety of sources to gather this information, such as YouTube channels, websites, and expert consultations. However, finding reliable, concise, and actionable advice can be time-consuming and overwhelming due to the vast amounts of content available online.
To address this issue, there is a need for an automated solution that can efficiently extract and organize relevant agricultural information from existing multimedia resources. YouTube, in particular, contains a wealth of expert-driven content, but users often face difficulties in navigating lengthy videos to find specific information. Additionally, there is no direct way to interact with this content to get personalized responses based on individual needs.

Solution:

This project aims to solve these challenges by developing an agriculture assistant chatbot that can extract scripts from YouTube channels, process the information using Retrieval-Augmented Generation (RAG), and provide interactive, customized responses to user queries. By leveraging this approach, the assistant can help users quickly access precise information on agricultural topics, ultimately saving time and improving decision-making in the field.

Required work:

To achieve the objectives of this project, performing the following tasks was required:

1) Extract YouTube videos scripts by providing only the channel name. If a script is available, it is extracted directly; otherwise, we use whisper model to generate the script.

2) Store extracted data in csv file and a vector database.

3) Use Rag technique to extract relevant information.

4) integrate an AI agent that customizes responses based on current weather conditions.

5) Create a simple interface using streamlit.

![Knowledge embeddings diagram](https://github.com/user-attachments/assets/cef40fbd-14e4-4ced-b0b1-ef58dac00c1f)

![Rag diagram](https://github.com/user-attachments/assets/b25e6a5c-efc3-4d72-acb8-4c9787599dc2)

![Capture d’écran 2024-10-20 165131](https://github.com/user-attachments/assets/63832b23-13e5-4325-a65d-c1a9901ce51f)

![chat_with_history](https://github.com/user-attachments/assets/e55023ba-c297-431b-aaac-278ec62a5955)

![chat_without_history](https://github.com/user-attachments/assets/dd939fcc-9763-4df8-8a34-86a627c1705b)

![references_and_spelling_mistakes](https://github.com/user-attachments/assets/dbe4e170-977c-4b22-8052-c51d2021f526)

