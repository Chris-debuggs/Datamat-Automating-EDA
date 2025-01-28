import json
import jsonschema
from jsonschema import validate

# Define schema for each JSON response
SCHEMAS = {
    "RESPONSE_JSON_EDA": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "object",
                "properties": {
                    "rows": {"type": "integer"},
                    "columns": {"type": "integer"},
                    "numerical_columns": {
                        "type": "object",
                        "patternProperties": {
                            ".*": {
                                "type": "object",
                                "properties": {
                                    "mean": {"type": "number"},
                                    "std": {"type": "number"},
                                    "min": {"type": "number"},
                                    "max": {"type": "number"}
                                },
                                "required": ["mean", "std", "min", "max"]
                            }
                        }
                    },
                    "categorical_columns": {
                        "type": "object",
                        "patternProperties": {
                            ".*": {
                                "type": "object",
                                "properties": {
                                    "unique_values": {"type": "array", "items": {"type": "string"}},
                                    "frequencies": {"type": "object"}
                                },
                                "required": ["unique_values", "frequencies"]
                            }
                        }
                    },
                    "missing_values": {"type": "object"},
                    "duplicates": {"type": "integer"}
                },
                "required": ["rows", "columns", "numerical_columns", "categorical_columns", "missing_values", "duplicates"]
            }
        },
        "required": ["summary"]
    },
    # Add schemas for other JSON responses like RESPONSE_JSON_CORR_PAT, etc.
}

# Function to validate JSON response against schema
def validate_json(template_name, json_file, schema_name):
    try:
        # Load JSON file
        with open(json_file, "r") as file:
            data = json.load(file)
        
        # Validate against schema
        validate(instance=data, schema=SCHEMAS[schema_name])
        print(f"✅ {template_name}: JSON file '{json_file}' is valid.")
    except jsonschema.exceptions.ValidationError as e:
        print(f"❌ {template_name}: Validation error in '{json_file}': {e.message}")
    except Exception as e:
        print(f"❌ {template_name}: Error loading or validating '{json_file}': {str(e)}")

if __name__ == "__main__":
    validate_json("TEMPLATE_EDA", "response_json_eda.json", "RESPONSE_JSON_EDA")
