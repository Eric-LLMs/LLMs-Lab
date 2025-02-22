# Introduction

![img.png](llm-developing-assistantgpt%2Fassets%2Fimg.png)  
![img_1.png](llm-developing-assistantgpt%2Fassets%2Fimg_1.png)  

# Running

1. Create a virtual environment and install dependencies:
   - `conda create -n py38_AssistantGPT python=3.8`
   - `conda activate py38_AssistantGPT`
   - `pip install -r requirements.txt`

2. Configure environment variables:
   - Open the `.env.example` file
   - Complete the `OPENAI_API_KEY`, `HTTP_PROXY`, and `HTTPS_PROXY` environment variables in the file
   - Rename the `.env.example` file to `.env`

3. Run the `qdtant` container with Docker

4. Run the `app.py` file

Example terminal output when successfully started:
```bash
> python app.py
Running on local URL:  http://127.0.0.1:7860

To create a public link, set `share=True` in `launch()`.
```

## Running Logs

### Uploading Files

Successful upload example:

```bash
2025-02-21 14:40:51.643 | TRACE    | __main__:fn_upload_files:172 - Component Input | unuploaded_file_paths: ['C:\\Users\\92047\\AppData\\Local\\Temp\\gradio\\2def4c9f9f49eebf9d276d42bf0badb385ce439d\\sample-pdf.pdf']
2025-02-21 14:40:51.644 | DEBUG    | utils:upload_files:86 - Input Parameter | file_path: C:\Users\92047\AppData\Local\Temp\gradio\2def4c9f9f49eebf9d276d42bf0badb385ce439d\sample-pdf.pdf <class 'str'>
2025-02-21 14:40:51.644 | TRACE    | utils:upload_files:100 - File is allowed to be processed | file_path: C:\Users\92047\AppData\Local\Temp\gradio\2def4c9f9f49eebf9d276d42bf0badb385ce439d\sample-pdf.pdf
2025-02-21 14:40:51.644 | DEBUG    | utils:upload_files:107 - File Info | file_name: sample-pdf.pdf, file_extension: .pdf, file_md5: e41ab92c3f938ddb3e82110becbbce3e
2025-02-21 14:40:52.162 | SUCCESS  | db_qdrant:get_points_count:43 - The collection already exists in the database | collection_name: e41ab92c3f938ddb3e82110becbbce3e points_count: 1
```