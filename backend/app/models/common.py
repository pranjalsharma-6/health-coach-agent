"""Shared model plumbing for Mongo documents."""

from datetime import datetime, timezone
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

# Mongo's ObjectId isn't JSON-serialisable and Pydantic doesn't know it.
# Coerce it to a string on the way in; the API only ever speaks strings.
PyObjectId = Annotated[str, BeforeValidator(lambda v: str(v))]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class MongoModel(BaseModel):
    """Base for models that map to a Mongo document."""

    model_config = ConfigDict(
        populate_by_name=True,
        use_enum_values=True,
        arbitrary_types_allowed=True,
    )

    id: PyObjectId | None = Field(default=None, alias="_id")

    @classmethod
    def from_mongo(cls, doc: dict[str, Any] | None):
        """Build a model from a raw Mongo document, or None if there isn't one."""
        if not doc:
            return None
        doc = dict(doc)
        if "_id" in doc:
            doc["_id"] = str(doc["_id"])
        return cls(**doc)

    def to_mongo(self, exclude_id: bool = True) -> dict[str, Any]:
        """Serialise for insertion. Drops the id so Mongo assigns one."""
        data = self.model_dump(by_alias=True, exclude_none=True)
        if exclude_id:
            data.pop("_id", None)
        return data
