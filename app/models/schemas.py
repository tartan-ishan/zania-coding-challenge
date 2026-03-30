from pydantic import BaseModel, Field


class StructuredAnswer(BaseModel):
    answer: str = Field(description="Direct answer to the question based solely on the provided context.")
    stepwise_reasoning: list[str] = Field(description="Ordered reasoning steps used to arrive at the answer.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0 and 1.")
    citations: list[str] = Field(description="Verbatim or near-verbatim excerpts from the context that support the answer.")


class QAResponse(BaseModel):
    answers: dict[str, StructuredAnswer]


class ErrorResponse(BaseModel):
    error_code: str
    message: str
