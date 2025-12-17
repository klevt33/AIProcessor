from typing import TypeAlias

from pydantic import BaseModel, Field, constr, field_validator

# Define a proper type alias
UNSPSCCode: TypeAlias = constr(pattern=r"^\d{8}$")  # type: ignore


# These get benefitted from LangChain’s automatic retry + JSON enforcement
class FinetunedLLMGeneralResponse(BaseModel):
    """
    You always get a valid PartInfo object, with UNSPSC="" if the value is bad.
    """

    ManufacturerName: str = Field(default="", description="Manufacturer name")
    PartNumber: str = Field(default="", description="Part number")
    # Either 8-digit string or empty string
    UNSPSC: str | UNSPSCCode = Field(default="", description="8 digit UNSPSC code as string, or empty string if unknown")

    @field_validator("UNSPSC", mode="before")
    def enforce_unspsc(cls, v):
        if isinstance(v, str) and v.isdigit() and len(v) == 8:
            return v
        return ""  # coerce invalid values to empty


class FinetunedLLMUNSPSCOnlyResponse(BaseModel):
    UNSPSC: str | UNSPSCCode = Field(default="", description="8 digit UNSPSC code as string, or empty string if unknown")

    @field_validator("UNSPSC", mode="before")
    def enforce_unspsc(cls, v):
        if isinstance(v, str) and v.isdigit() and len(v) == 8:
            return v
        return ""  # coerce invalid values to empty
