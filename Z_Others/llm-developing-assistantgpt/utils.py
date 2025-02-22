import time
import traceback

from db_qdrant import *
from AssistantGPT import AssistantGPT
from file_processor import FileProcessor
from file_processor_helper import FileProcessorHelper


def create_result_dict(code, msg=None, data=None):
    """
    生成一个包含代码、消息和数据的字典，用于表示函数执行的结果。

    Parameters:
    - code (int): 表示执行结果的代码，通常用于指示成功或失败等状态。
    - msg (str, optional): 包含有关执行结果的描述性消息，可以为空。
    - data (any type, optional): 附加数据，可以是任何类型的对象。

    Returns:
    dict: 包含代码、消息和数据的字典对象。

    Example:
    >>> result = create_result_dict(200, "操作成功", {'user_id': 123, 'username': 'John'})
    >>> print(result)
    {'code': 200, 'msg': '操作成功', 'data': {'user_id': 123, 'username': 'John'}}
    """
    result = {
        'code': code,
        'msg': msg,
        'data': data
    }
    return result


# 文件入向量数据库
def file_to_vectordb(file_path, file_name, file_extension, file_md5):

    # 集合名就是文件的 MD5 值
    collection_name = file_md5

    # 创建 Qdrant 类对象
    qdrant = Qdrant()

    # This line is for testing delete_collection
    # qdrant.client.delete_collection(collection_name=collection_name)

    # 获取集合里的数据数量 points_count，取值有三种情况: 0、>0、-1
    points_count = qdrant.get_points_count(collection_name)

    if points_count == 0:
        # case 1: 刚创建完集合，集合里没有节点
        # 创建 FileProcessorHelper 类对象
        file_processor_helper = FileProcessorHelper(
            file_path=file_path,
            file_name=file_name,
            file_extension=file_extension,
            file_md5=file_md5,
        )

        # 获取 docs
        docs = file_processor_helper.file_to_docs()
        logger.trace(f"docs: {docs}")

        # 切分 docs
        docs = file_processor_helper.split_docs(docs)
        texts = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]

        # 向量化 docs
        payloads = build_payloads(texts, metadatas)
        gpt = AssistantGPT()
        embeddings = gpt.get_embeddings(texts)
        # 插入节点
        if qdrant.add_points(collection_name, embeddings, payloads):
            return file_path
    elif points_count > 0:
        # case 2: 库里已有该集合，且该集合有节点
        return file_path
    else:  # 等价于 points_count == -1
        # case 3: `创建集合失败`或`获取集合信息时发生错误`
        return ''


def upload_files(file_path):
    try:
        # 打印输入参数
        logger.info(f"输入参数 | file_path: {file_path} {type(file_path)}")

        # 检查输入参数
        if not file_path:
            return create_result_dict(400, '没有上传文件')

        # 创建 FileProcessor 类对象
        file_processor = FileProcessor(file_path=file_path)

        # 检查文件是否允许处理
        if not file_processor.is_allowed_file():
            # 文件后缀不允许处理，直接返回
            return create_result_dict(400, f'暂不支持此文件后缀: {file_path}')

        logger.trace(f"文件允许被处理 | file_path: {file_path}")

        # 处理文件
        # 获取文件的更多信息
        file_name = file_processor.get_file_name()
        file_extension = file_processor.get_file_extension()
        file_md5 = file_processor.get_file_md5()
        logger.info(
            f"文件信息 | file_name: {file_name}, file_extension: {file_extension}, file_md5: {file_md5}")

        # 文件存入向量数据库
        uploaded_file_path = file_to_vectordb(
            file_path, file_name, file_extension, file_md5)

        # 处理成功
        if uploaded_file_path:
            # 处理成功
            return create_result_dict(
                200, None, {
                    'uploaded_file_path': uploaded_file_path
                })
        else:
            # 处理失败
            return create_result_dict(500)
    except Exception as e:
        # 打印完整错误信息
        error_str = traceback.format_exc()
        logger.error(error_str)
        # 处理失败
        return create_result_dict(500)


def build_context(qdrant, collection_names, question_vector, top_n):

    # 执行相似度搜索查询并获取 ScoredPoint 对象列表
    scored_points = []
    for collection_name in collection_names:
        scored_points_by_current_collection = qdrant.search(
            collection_name, question_vector, limit=top_n)
        scored_points.extend(scored_points_by_current_collection)

    # 将 ScoredPoint 对象列表转换为字典列表
    points = []
    for scored_point in scored_points:
        point = {
            "id": scored_point.id,
            "score": scored_point.score,
            "payload": scored_point.payload
        }
        points.append(point)

    # 字典列表按分数降序排序
    points.sort(key=lambda x: x['score'], reverse=True)
    points = points[:top_n]
    logger.trace(f"points: {points}")

    # 构建上下文
    contexts = []
    for point in points:
        context = point['payload']['page_content']
        # metadata = point['payload']['metadata']
        contexts.append(context)
    context = "\n---\n".join(contexts)

    return context

# 构建文档问答 prompt
def build_chat_document_prompt(file_paths, user_input, chat_history, top_n):
    try:
        # 打印参数
        logger.debug(
            f"file_paths: {file_paths}, user_input: {user_input}, chat_history: {chat_history}, top_n: {top_n}")

        # qdrant 参数
        qdrant = Qdrant()

        # collection_names 参数
        collection_names = []
        for file_path in file_paths:
            file_bytes = FileProcessor.get_file_bytes(file_path)
            file_md5 = FileProcessor.calculate_md5(file_bytes)
            collection_names.append(file_md5)
        logger.debug(f"collection_names: {collection_names}")

        # question_vector 参数
        gpt = AssistantGPT()
        question_vectors = retry(gpt.get_embeddings, args=([user_input]))
        if not question_vectors:
            logger.error("获取 question_vector 参数失败")
            return ''
        question_vector = question_vectors[0]

        # context
        top_n = int(top_n)
        context = build_context(
            qdrant,
            collection_names,
            question_vector,
            top_n)
        logger.trace(f"context: \n{context}")

        # chat_history_str
        chat_history_str = ""
        for chat in chat_history[:-1]:
            # 检查是否是用户消息
            if chat[0]:
                chat_history_str += f'user:{chat[0]}\n'
            if chat[1]:
                chat_history_str += f'assistant:{chat[1]}\n'
        chat_history_str = chat_history_str[:-1]  # 去掉最后一个'\n'
        logger.trace(f"chat_history_str: \n{chat_history_str}")

        # 构建 prompt
        prompt = f"""你是一位文档问答助手，你会基于`文档内容`和`对话历史`回答user的问题。如果用户的问题与`文档内容`无关，就不用强行根据`文档内容`回答。

文档内容：```
{context}```

对话历史：```
{chat_history_str}```

user: ```{user_input}```
assistant: """
        logger.info(f"prompt: \n{prompt}")
        return prompt
    except Exception as e:
        error_str = traceback.format_exc()
        logger.error(error_str)
        return ''


def retry(func, args=None, kwargs=None, retries=3, delay=1):
    """
    重试机制函数
    :param func: 需要重试的函数
    :param args: 函数参数，以元组形式传入
    :param kwargs: 函数关键字参数，以字典形式传入
    :param retries: 重试次数，默认为3
    :param delay: 重试间隔时间，默认为1秒
    :return: 函数执行结果
    """
    for i in range(retries):
        try:
            if args is None and kwargs is None:
                result = func()
            elif args is not None and kwargs is None:
                result = func(*args)
            elif args is None and kwargs is not None:
                result = func(**kwargs)
            else:
                result = func(*args, **kwargs)
            return result  # 如果函数执行成功，返回结果
        except Exception as e:
            logger.warning(f"{func.__name__}函数第{i + 1}次重试：{e}")
            time.sleep(delay)
    logger.error(f"{func.__name__}函数重试次数已用完")


def build_payloads(texts, metadatas):
    payloads = [
        {
            "page_content": text,
            "metadata": metadata,
        }
        for text, metadata in zip(texts, metadatas)
    ]
    return payloads
