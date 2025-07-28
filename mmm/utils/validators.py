import pandera as pa
from pandera import Check, Column, DataFrameSchema

# Define schema for the merged dataset
merged_data_schema = DataFrameSchema(
    {
        "date": Column(pa.DateTime),
        # sales must be present, so dropping rows with missing sales is safer.
        "sales": Column(pa.Float, Check.ge(0), nullable=False),
        "tv_spend": Column(pa.Float, Check.ge(0)),
        "radio_spend": Column(pa.Float, Check.ge(0)),
        "social_media_spend": Column(pa.Float, Check.ge(0)),
        "search_spend": Column(pa.Float, Check.ge(0)),
        "print_spend": Column(pa.Float, Check.ge(0)),
        "outdoor_spend": Column(pa.Float, Check.ge(0)),
    },
    strict=True,
    coerce=True,  # Ensures correct dtypes
)
