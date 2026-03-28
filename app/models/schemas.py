from pydantic import BaseModel


class QAResponse(BaseModel):
    answers: dict[str, str]


class ErrorResponse(BaseModel):
    detail: str
