from pydantic import BaseModel

class SubmitMuinmosAnswerRequest(BaseModel):
    assessment_id: str
    answer: list

class CreateMuinmosAssessmentByGuestAccountRequest(BaseModel):
    order_code: str
