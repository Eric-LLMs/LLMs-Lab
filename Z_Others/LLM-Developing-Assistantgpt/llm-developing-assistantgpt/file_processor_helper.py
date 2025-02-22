import os
from typing import List

import pdfplumber
import tiktoken

from config import CHUNK_OVERLAP, CHUNK_SIZE
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


class FileProcessorHelper:
    def __init__(
            self,
            file_path: str,
            file_name: str = None,
            file_extension: str = None,
            file_md5: str = None,
    ):
        self.file_path = file_path
        self.file_name = file_name
        self.file_extension = file_extension
        self.file_md5 = file_md5

    # 获取docs
    def file_to_docs(self) -> List:

        strategy_mapping = {
            '.pdf': self.pdf_file_to_docs,
            '.txt': self.txt_file_to_docs,
            # '.doc': self.word_file_to_docs,
            # '.docx': self.word_file_to_docs,
            # '.md': self.md_file_to_docs,
            # '.xls': self.excel_file_to_docs,
            # '.xlsx': self.excel_file_to_docs,
            # '.csv': self.csv_file_to_docs,
            # '.jpg': self.image_file_to_docs,
            # '.jpeg': self.image_file_to_docs,
            # '.png': self.image_file_to_docs,
            # '.gif': self.image_file_to_docs,
            # '.ico': self.image_file_to_docs,
            # '.svg': self.image_file_to_docs,
            # '.bmp': self.image_file_to_docs,
            # '.ppt': self.ppt_file_to_docs,
            # '.pptx': self.ppt_file_to_docs,
            # '.zip': self.zip_file_to_docs,
            # '.mp3': self.audio_file_to_docs,
            # '.wav': self.audio_file_to_docs,
            # '.mp4': self.video_file_to_docs,
        }
        func = strategy_mapping.get(self.file_extension)
        return func(self.file_path)

    # 切分docs
    def split_docs(self, docs):
        # 切分
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=self.tiktoken_len,
        )
        texts = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]
        docs = text_splitter.create_documents(texts, metadatas=metadatas)
        return docs

    @staticmethod
    def pdf_file_to_docs(file_path: str) -> List[Document]:
        file_name = os.path.basename(file_path)

        docs = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    doc = Document(
                        page_content=page_text,
                        metadata=dict(
                            {
                                "file_name": file_name,
                                "page": page.page_number,
                                "total_pages": len(pdf.pages),
                            },
                            **{
                                k: pdf.metadata[k]
                                for k in pdf.metadata
                                if isinstance(pdf.metadata[k], (str, int))
                            },
                        ),
                    )
                    docs.append(doc)
        return docs

    @staticmethod
    def txt_file_to_docs(file_path: str) -> List[Document]:
        file_name = os.path.basename(file_path)

        with open(file_path, 'r') as file:
            text = file.read()
        if not text:
            return []
        return [
            Document(
                page_content=text,
                metadata={
                    "file_name": file_name})]

    @staticmethod
    def tiktoken_len(text, model="gpt-3.5-turbo"):
        """
        https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        :param text:
        :param model:
        :return:
        """
        # use cl100k_base tokenizer for gpt-3.5-turbo and gpt-4
        encoding = tiktoken.get_encoding('cl100k_base')
        # encoding = tiktoken.encoding_for_model(model)
        tokens = encoding.encode(
            text,
            disallowed_special=()  # 禁用对所有特殊标记的检查
        )
        return len(tokens)


if __name__ == "__main__":
    # 测试
    file_path = "assets\\sample-pdf.pdf"
    file_name = "sample-pdf.pdf"
    file_extension = ".pdf"
    file_md5 = "e41ab92c3f938ddb3e82110becbbce3e"

    file_processor_helper = FileProcessorHelper(
        file_path=file_path,
        file_name=file_name,
        file_extension=file_extension,
        file_md5=file_md5,
    )
    docs = file_processor_helper.file_to_docs()
    print(docs)
    # [Document(page_content='样本\nPDF\n将本地 PDF 文件直接拖到此窗口中，或单击此页面右上角的第一个按钮上传本地 PDF\n文件，沉浸式翻译扩展将立即开始翻译您的 PDF 文件。\n翻译结果将显示在页面右侧，因此您可以轻松地交叉引用原始 PDF 文件。\n让我们一起享受身临其境的翻译体验吧！\n笔记：\n1. 沉浸式翻译扩展支持导出双语PDF，翻译完成后请点击右上角的保存按钮下载双语\nPDF。\n2. 如果PDF页面是图片，则翻译后的页面将是空白页，因为我们只翻译了文本\n部分。\n3. 某些PDF文档可能有重叠的翻译，如果发生这种情况，您可以选择翻译的文本框，然\n后拖动、缩放或删除文本框以获得更好的阅读体验。如下图所示：\n4. 快捷方式控制样式按钮提供了一种方便的方式来控制翻译等：', metadata={'file_name': 'sample-pdf.pdf', 'page': 1, 'total_pages': 2, 'Creator': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36', 'Producer': 'Skia/PDF m119', 'CreationDate': "D:20231201072920+00'00'", 'ModDate': "D:20231201072920+00'00'"})]

    # 测试 tiktoken_len
    print(FileProcessorHelper.tiktoken_len("你好\n你好有什么能帮你的吗？"))