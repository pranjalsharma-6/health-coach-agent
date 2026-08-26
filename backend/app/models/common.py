"""Shared model plumbing for Mongo documents."""

from datetime import date, datetime, timezone
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
        return _bson_safe(data)


def _bson_safe(value: Any) -> Any:
    """Recursively convert values BSON can't encode.

    BSON has no bare-date type — only datetime — so a `datetime.date` anywhere
    in a document (a nested adherence snapshot, say) raises on insert. Dates
    become ISO strings, which also sort correctly, matching how `log_date` is
    stored and queried.

    `datetime` subclasses `date`, so the order of these checks matters.
    """
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _bson_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_bson_safe(item) for item in value]
    return value
