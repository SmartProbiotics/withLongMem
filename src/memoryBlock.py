from pydantic import BaseModel
class MemoryBlock(BaseModel):
    text:str
    metadata:dict
    """ :Memory Block:
        'text': "内存块文本",
        'metadata':
            'title': "内存块标题",
            'source': "内存块来源"
    """
    