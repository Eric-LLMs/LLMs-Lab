import hashlib
import os
from typing import Any, Dict, List, Union


class FileProcessor:
    # 定义允许处理的文件后缀列表，作为类属性
    ALLOWED_EXTENSIONS = ['.txt', '.pdf']

    def __init__(self, file_path):
        self.file_path = file_path

    # 获取文件后缀
    def get_file_extension(self):
        _, file_extension = os.path.splitext(self.file_path)
        return file_extension.lower()  # 转换为小写以方便比较

    def is_allowed_file(self):
        # 获取文件后缀
        file_extension = self.get_file_extension()

        # 检查文件后缀是否在允许处理的列表中
        return file_extension in self.ALLOWED_EXTENSIONS

    # 获取文件名（不包含后缀）
    def get_file_name(self):
        file_name = os.path.basename(self.file_path)
        return file_name

    # 获取文件MD5值
    # todo @staticmethod
    def get_file_md5(self):
        file_bytes = self.get_file_bytes(self.file_path)
        file_md5 = self.calculate_md5(file_bytes)
        return file_md5

    @staticmethod
    def get_file_bytes(file_path: str):
        # 打开文件
        with open(file_path, 'rb') as file:
            # 读取文件内容为字节流
            file_bytes = file.read()

        return file_bytes

    # 计算输入数据的MD5哈希值
    @staticmethod
    def calculate_md5(input_data: Union[str, bytes]) -> str:
        # 创建一个MD5对象
        md5 = hashlib.md5()

        # 判断输入是字符串还是字节流
        if isinstance(input_data, str):
            # 如果是字符串，将其编码为字节流再更新MD5对象的内容
            md5.update(input_data.encode('utf-8'))
        elif isinstance(input_data, bytes):
            # 如果已经是字节流，直接更新MD5对象的内容
            md5.update(input_data)
        else:
            raise ValueError("Input data must be either a string or bytes")
        return md5.hexdigest()  # 获取MD5哈希值，并以十六进制字符串的形式返回

    # # 文件入向量数据库
    # def process_file(self) -> Dict[str, Union[int, str, Dict]]:
    #     try:
    #         file_name = self.get_file_name()
    #         file_extension = self.get_file_extension()
    #         file_md5 = self.get_file_md5()
    #         collection_name = file_md5
    #
    #         qdrant = Qdrant()
    #
    #         # this line just for test： 先删除再存储
    #         # qdrant.client.delete_collection(collection_name=collection_name)
    #
    #         # 获取集合里的数据数量 points_count，取值有三种情况: 0、>0、-1
    #         points_count = qdrant.get_points_count(collection_name)
    #         logger.trace(f"points_count: {points_count}")
    #
    #         if points_count == 0:
    #             # case 1: 刚创建完集合，集合里没有节点
    #             # 辅助类对象
    #             file_processor_helper = FileProcessorHelper(
    #                 file_path=self.file_path,
    #                 file_name=file_name,
    #                 file_extension=file_extension,
    #                 file_md5=file_md5,
    #             )
    #             # 获取docs
    #             docs = file_processor_helper.file_to_docs()
    #             logger.debug(f"docs: {docs}")
    #             # 切分docs
    #             docs = file_processor_helper.split_docs(docs)
    #             texts = [doc.page_content for doc in docs]
    #             metadatas = [doc.metadata for doc in docs]
    #             # 向量化docs
    #             payloads = build_payloads(texts, metadatas)
    #             gpt = AssistantGPT()
    #             embeddings = gpt.get_embeddings(texts)
    #             # 插入节点
    #             if qdrant.add_points(collection_name, embeddings, payloads):
    #                 result = custom_jsonify(201, None, {'file_path': self.file_path})
    #         elif points_count > 0:
    #             # case 2: 库里已有该集合，且该集合有节点
    #             result = custom_jsonify(202, None, {'file_path': self.file_path})
    #         else:  # 等价于 points_count == -1
    #             # case 3: `创建集合失败`或`获取集合信息时发生错误`
    #             logger.error("向量数据库创建集合时失败")
    #             result = custom_jsonify(501)
    #     except Exception as e:
    #         logger.error(e)
    #         result = custom_jsonify(502)
    #     return result


if __name__ == "__main__":
    # 示例用法
    file_path = "uploads/1/simple-pdf.pdf"  # 替换为实际的文件路径
    processor = FileProcessor(file_path)
    result = processor.is_allowed_file()
    print(result)  # 输出 True 或 False
