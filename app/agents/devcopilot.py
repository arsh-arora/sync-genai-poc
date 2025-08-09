"""
Developer & Partner API Copilot
Interactive guide for partner APIs with code snippets, validation, and Postman collections
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Literal
import jsonschema
from jsonschema import validate, ValidationError

from pydantic import BaseModel, Field

from app.llm.gemini import chat

logger = logging.getLogger(__name__)

# Pydantic models for strict input/output schema enforcement
class ValidationError(BaseModel):
    path: str
    msg: str

class ValidationResult(BaseModel):
    ok: bool
    errors: List[ValidationError] = []

class CodeSnippet(BaseModel):
    lang: str
    code: str

class PostmanInfo(BaseModel):
    filename: str

class DevCopilotResponse(BaseModel):
    endpoint: str
    snippet: CodeSnippet
    example_request: Dict[str, Any]
    validation: ValidationResult
    postman: PostmanInfo

class DevCopilot:
    """
    Developer & Partner API Copilot for interactive API integration guidance
    """
    
    def __init__(self):
        """Initialize DevCopilot with OpenAPI registry"""
        self.openapi_dir = Path("app/openapi")
        self.downloads_dir = Path("app/ui/downloads")
        self.downloads_dir.mkdir(exist_ok=True)
    
    def generate_code_guide(
        self, 
        service: str, 
        endpoint: Optional[str] = None, 
        lang: Literal["python", "javascript", "java", "curl"] = "python",
        sample: Optional[Dict[str, Any]] = None
    ) -> DevCopilotResponse:
        """
        Main processing pipeline for API code generation and guidance
        
        Args:
            service: Service name to load OpenAPI spec for
            endpoint: Specific endpoint (if None, suggests top 3)
            lang: Programming language for code snippet
            sample: Sample payload to validate
            
        Returns:
            DevCopilotResponse with code snippet, validation, and Postman collection
        """
        try:
            logger.info(f"Generating code guide for service: {service}")
            
            # Step 1: Load OpenAPI spec
            spec = self.openapi_registry(service)
            if not spec:
                return self._error_response(f"Service '{service}' not found in registry")
            
            # Step 2: Determine endpoint
            if not endpoint:
                suggested_endpoints = self._suggest_endpoints(spec, service)
                endpoint = suggested_endpoints[0] if suggested_endpoints else None
                
            if not endpoint:
                return self._error_response("No suitable endpoint found")
            
            logger.info(f"Using endpoint: {endpoint}")
            
            # Step 3: Generate code snippet
            snippet = self._generate_code_snippet(spec, endpoint, lang)
            
            # Step 4: Create example request
            example_request = self._generate_example_request(spec, endpoint)
            
            # Step 5: Validate sample payload if provided
            validation = ValidationResult(ok=True, errors=[])
            if sample:
                validation = self.payload_validate(sample, spec, endpoint)
            
            # Step 6: Generate Postman collection
            postman_info = self.postman_export(spec, endpoint)
            
            return DevCopilotResponse(
                endpoint=endpoint,
                snippet=snippet,
                example_request=example_request,
                validation=validation,
                postman=postman_info
            )
            
        except Exception as e:
            logger.error(f"Error generating code guide: {e}")
            return self._error_response(f"Error: {str(e)}")
    
    def openapi_registry(self, service: str) -> Optional[Dict[str, Any]]:
        """
        Load OpenAPI specification from registry
        
        Args:
            service: Service name
            
        Returns:
            OpenAPI spec dictionary or None if not found
        """
        try:
            spec_file = self.openapi_dir / f"{service}.json"
            if not spec_file.exists():
                logger.error(f"OpenAPI spec not found: {spec_file}")
                return None
            
            with open(spec_file, 'r') as f:
                spec = json.load(f)
            
            logger.info(f"Loaded OpenAPI spec for {service}")
            return spec
            
        except Exception as e:
            logger.error(f"Error loading OpenAPI spec for {service}: {e}")
            return None
    
    def payload_validate(
        self, 
        payload: Dict[str, Any], 
        spec: Dict[str, Any], 
        endpoint: str
    ) -> ValidationResult:
        """
        Validate payload against OpenAPI schema
        
        Args:
            payload: JSON payload to validate
            spec: OpenAPI specification
            endpoint: API endpoint path
            
        Returns:
            ValidationResult with errors if any
        """
        try:
            # Find the schema for the endpoint
            schema = self._extract_request_schema(spec, endpoint)
            if not schema:
                return ValidationResult(
                    ok=False, 
                    errors=[ValidationError(path="", msg="No schema found for endpoint")]
                )
            
            # Validate payload against schema
            try:
                validate(instance=payload, schema=schema)
                return ValidationResult(ok=True, errors=[])
            
            except jsonschema.ValidationError as e:
                errors = self._format_validation_errors([e])
                return ValidationResult(ok=False, errors=errors)
            
            except jsonschema.SchemaError as e:
                return ValidationResult(
                    ok=False,
                    errors=[ValidationError(path="schema", msg=f"Invalid schema: {str(e)}")]
                )
                
        except Exception as e:
            logger.error(f"Error validating payload: {e}")
            return ValidationResult(
                ok=False,
                errors=[ValidationError(path="", msg=f"Validation error: {str(e)}")]
            )
    
    def mock_sandbox_call(self, endpoint: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mock sandbox call that echoes request with basic constraints
        
        Args:
            endpoint: API endpoint
            request: Request payload
            
        Returns:
            Mock response with status and body
        """
        try:
            # Basic mock response based on endpoint
            if "payment" in endpoint.lower():
                return {
                    "status": 201,
                    "body": {
                        "id": "pay_mock_123456",
                        "amount": request.get("amount", 100.00),
                        "currency": request.get("currency", "USD"),
                        "status": "completed",
                        "createdAt": "2024-01-01T12:00:00Z"
                    }
                }
            elif "balance" in endpoint.lower():
                return {
                    "status": 200,
                    "body": {
                        "accountId": request.get("accountId", "acc_123"),
                        "availableBalance": 1500.00,
                        "currentBalance": 2000.00,
                        "currency": "USD"
                    }
                }
            else:
                return {
                    "status": 200,
                    "body": {
                        "message": "Mock response",
                        "request": request
                    }
                }
                
        except Exception as e:
            return {
                "status": 500,
                "body": {
                    "error": f"Mock error: {str(e)}"
                }
            }
    
    def postman_export(
        self, 
        spec: Dict[str, Any], 
        focus_endpoint: Optional[str] = None
    ) -> PostmanInfo:
        """
        Export Postman collection for the API spec
        
        Args:
            spec: OpenAPI specification
            focus_endpoint: Specific endpoint to focus on
            
        Returns:
            PostmanInfo with filename
        """
        try:
            service_name = spec.get("info", {}).get("title", "API").replace(" ", "_")
            filename = f"{service_name}_collection.json"
            filepath = self.downloads_dir / filename
            
            # Create basic Postman collection
            collection = {
                "info": {
                    "name": spec.get("info", {}).get("title", "API Collection"),
                    "description": spec.get("info", {}).get("description", ""),
                    "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
                },
                "auth": {
                    "type": "apikey",
                    "apikey": [
                        {
                            "key": "key",
                            "value": "X-API-Key",
                            "type": "string"
                        },
                        {
                            "key": "value",
                            "value": "{{API_KEY}}",
                            "type": "string"
                        }
                    ]
                },
                "variable": [
                    {
                        "key": "baseUrl",
                        "value": spec.get("servers", [{}])[0].get("url", ""),
                        "type": "string"
                    }
                ],
                "item": []
            }
            
            # Add requests for each endpoint
            paths = spec.get("paths", {})
            for path, methods in paths.items():
                if focus_endpoint and path != focus_endpoint:
                    continue
                    
                for method, operation in methods.items():
                    if method.upper() in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                        request_item = self._create_postman_request(
                            path, method.upper(), operation, spec
                        )
                        collection["item"].append(request_item)
            
            # Write collection to file
            with open(filepath, 'w') as f:
                json.dump(collection, f, indent=2)
            
            logger.info(f"Exported Postman collection: {filename}")
            return PostmanInfo(filename=filename)
            
        except Exception as e:
            logger.error(f"Error exporting Postman collection: {e}")
            return PostmanInfo(filename="error_collection.json")
    
    def _suggest_endpoints(self, spec: Dict[str, Any], service: str) -> List[str]:
        """Suggest top 3 endpoints using Gemini"""
        try:
            paths = list(spec.get("paths", {}).keys())
            if len(paths) <= 3:
                return paths
            
            # Use Gemini to suggest most relevant endpoints
            system_prompt = f"""You are an API expert. Given the following API endpoints for the {service} service, suggest the top 3 most commonly used endpoints that developers would want to integrate first.

Consider endpoints that are:
1. Core functionality (create, read operations)
2. Most likely to be used in initial integration
3. Provide immediate value to developers

Return only the endpoint paths as a JSON array, nothing else."""

            user_message = f"API endpoints: {json.dumps(paths)}"
            messages = [{"role": "user", "content": user_message}]
            
            response = chat(messages, system=system_prompt)
            
            try:
                suggested = json.loads(response.strip())
                return suggested[:3] if isinstance(suggested, list) else paths[:3]
            except:
                return paths[:3]
                
        except Exception as e:
            logger.error(f"Error suggesting endpoints: {e}")
            return list(spec.get("paths", {}).keys())[:3]
    
    def _generate_code_snippet(
        self, 
        spec: Dict[str, Any], 
        endpoint: str, 
        lang: str
    ) -> CodeSnippet:
        """Generate code snippet for the endpoint"""
        try:
            base_url = spec.get("servers", [{}])[0].get("url", "https://api.example.com")
            
            # Get endpoint details
            path_info = spec.get("paths", {}).get(endpoint, {})
            method = list(path_info.keys())[0] if path_info else "get"
            operation = path_info.get(method, {})
            
            # Generate example request body
            example_body = self._generate_example_request(spec, endpoint)
            
            if lang == "python":
                code = self._generate_python_snippet(base_url, endpoint, method, example_body)
            elif lang == "javascript":
                code = self._generate_javascript_snippet(base_url, endpoint, method, example_body)
            elif lang == "java":
                code = self._generate_java_snippet(base_url, endpoint, method, example_body)
            elif lang == "curl":
                code = self._generate_curl_snippet(base_url, endpoint, method, example_body)
            else:
                code = f"# Code snippet for {lang} not implemented yet"
            
            return CodeSnippet(lang=lang, code=code)
            
        except Exception as e:
            logger.error(f"Error generating code snippet: {e}")
            return CodeSnippet(
                lang=lang, 
                code=f"# Error generating {lang} snippet: {str(e)}"
            )
    
    def _generate_python_snippet(
        self, 
        base_url: str, 
        endpoint: str, 
        method: str, 
        example_body: Dict[str, Any]
    ) -> str:
        """Generate Python code snippet"""
        if method.upper() == "GET":
            return f'''import requests

# API Configuration
API_KEY = "your_api_key_here"
BASE_URL = "{base_url}"

# Headers
headers = {{
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}}

# Make GET request
response = requests.get(
    f"{{BASE_URL}}{endpoint}",
    headers=headers
)

# Handle response
if response.status_code == 200:
    data = response.json()
    print("Success:", data)
else:
    print(f"Error {{response.status_code}}: {{response.text}}")
'''
        else:
            return f'''import requests
import json

# API Configuration
API_KEY = "your_api_key_here"
BASE_URL = "{base_url}"

# Headers
headers = {{
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}}

# Request payload
payload = {json.dumps(example_body, indent=4)}

# Make {method.upper()} request
response = requests.{method.lower()}(
    f"{{BASE_URL}}{endpoint}",
    headers=headers,
    json=payload
)

# Handle response
if response.status_code in [200, 201]:
    data = response.json()
    print("Success:", data)
else:
    print(f"Error {{response.status_code}}: {{response.text}}")
'''
    
    def _generate_javascript_snippet(
        self, 
        base_url: str, 
        endpoint: str, 
        method: str, 
        example_body: Dict[str, Any]
    ) -> str:
        """Generate JavaScript code snippet"""
        if method.upper() == "GET":
            return f'''// API Configuration
const API_KEY = "your_api_key_here";
const BASE_URL = "{base_url}";

// Make GET request
fetch(`${{BASE_URL}}{endpoint}`, {{
    method: 'GET',
    headers: {{
        'X-API-Key': API_KEY,
        'Content-Type': 'application/json'
    }}
}})
.then(response => {{
    if (!response.ok) {{
        throw new Error(`HTTP error! status: ${{response.status}}`);
    }}
    return response.json();
}})
.then(data => {{
    console.log('Success:', data);
}})
.catch(error => {{
    console.error('Error:', error);
}});
'''
        else:
            return f'''// API Configuration
const API_KEY = "your_api_key_here";
const BASE_URL = "{base_url}";

// Request payload
const payload = {json.dumps(example_body, indent=2)};

// Make {method.upper()} request
fetch(`${{BASE_URL}}{endpoint}`, {{
    method: '{method.upper()}',
    headers: {{
        'X-API-Key': API_KEY,
        'Content-Type': 'application/json'
    }},
    body: JSON.stringify(payload)
}})
.then(response => {{
    if (!response.ok) {{
        throw new Error(`HTTP error! status: ${{response.status}}`);
    }}
    return response.json();
}})
.then(data => {{
    console.log('Success:', data);
}})
.catch(error => {{
    console.error('Error:', error);
}});
'''
    
    def _generate_curl_snippet(
        self, 
        base_url: str, 
        endpoint: str, 
        method: str, 
        example_body: Dict[str, Any]
    ) -> str:
        """Generate cURL code snippet"""
        if method.upper() == "GET":
            return f'''curl -X GET "{base_url}{endpoint}" \\
  -H "X-API-Key: your_api_key_here" \\
  -H "Content-Type: application/json"
'''
        else:
            return f'''curl -X {method.upper()} "{base_url}{endpoint}" \\
  -H "X-API-Key: your_api_key_here" \\
  -H "Content-Type: application/json" \\
  -d '{json.dumps(example_body)}'
'''
    
    def _generate_java_snippet(
        self, 
        base_url: str, 
        endpoint: str, 
        method: str, 
        example_body: Dict[str, Any]
    ) -> str:
        """Generate Java code snippet"""
        return f'''import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.URI;

public class APIClient {{
    private static final String API_KEY = "your_api_key_here";
    private static final String BASE_URL = "{base_url}";
    
    public static void main(String[] args) throws Exception {{
        HttpClient client = HttpClient.newHttpClient();
        
        HttpRequest.Builder requestBuilder = HttpRequest.newBuilder()
            .uri(URI.create(BASE_URL + "{endpoint}"))
            .header("X-API-Key", API_KEY)
            .header("Content-Type", "application/json");
            
        {"// Request body" if method.upper() != "GET" else ""}
        {"String requestBody = " + json.dumps(json.dumps(example_body)) + ";" if method.upper() != "GET" else ""}
        
        HttpRequest request = requestBuilder
            .{method.upper()}({f"HttpRequest.BodyPublishers.ofString(requestBody)" if method.upper() != "GET" else "HttpRequest.BodyPublishers.noBody()"})
            .build();
            
        HttpResponse<String> response = client.send(request, 
            HttpResponse.BodyHandlers.ofString());
            
        System.out.println("Status: " + response.statusCode());
        System.out.println("Response: " + response.body());
    }}
}}
'''
    
    def _generate_example_request(self, spec: Dict[str, Any], endpoint: str) -> Dict[str, Any]:
        """Generate example request payload"""
        try:
            path_info = spec.get("paths", {}).get(endpoint, {})
            
            # Find POST/PUT method with request body
            for method, operation in path_info.items():
                request_body = operation.get("requestBody", {})
                if request_body:
                    content = request_body.get("content", {})
                    json_content = content.get("application/json", {})
                    schema = json_content.get("schema", {})
                    
                    return self._generate_example_from_schema(schema, spec)
            
            # If no request body, return empty dict
            return {}
            
        except Exception as e:
            logger.error(f"Error generating example request: {e}")
            return {"error": "Could not generate example"}
    
    def _generate_example_from_schema(
        self, 
        schema: Dict[str, Any], 
        spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate example data from JSON schema"""
        try:
            # Handle $ref
            if "$ref" in schema:
                ref_path = schema["$ref"].split("/")
                ref_schema = spec
                for part in ref_path[1:]:  # Skip '#'
                    ref_schema = ref_schema.get(part, {})
                return self._generate_example_from_schema(ref_schema, spec)
            
            schema_type = schema.get("type", "object")
            
            if schema_type == "object":
                example = {}
                properties = schema.get("properties", {})
                required = schema.get("required", [])
                
                for prop_name, prop_schema in properties.items():
                    if prop_name in required or len(example) < 3:  # Include required + some optional
                        example[prop_name] = self._generate_example_value(prop_schema, spec)
                
                return example
            
            elif schema_type == "array":
                items_schema = schema.get("items", {})
                return [self._generate_example_from_schema(items_schema, spec)]
            
            else:
                return self._generate_example_value(schema, spec)
                
        except Exception as e:
            logger.error(f"Error generating example from schema: {e}")
            return {"example": "value"}
    
    def _generate_example_value(self, schema: Dict[str, Any], spec: Dict[str, Any]) -> Any:
        """Generate example value based on schema type"""
        schema_type = schema.get("type", "string")
        
        if schema_type == "string":
            if schema.get("format") == "date-time":
                return "2024-01-01T12:00:00Z"
            elif "enum" in schema:
                return schema["enum"][0]
            else:
                return "example_string"
        
        elif schema_type == "number":
            minimum = schema.get("minimum", 0)
            return max(100.0, minimum + 1)
        
        elif schema_type == "integer":
            minimum = schema.get("minimum", 0)
            return max(1, minimum + 1)
        
        elif schema_type == "boolean":
            return True
        
        elif schema_type == "object":
            return self._generate_example_from_schema(schema, spec)
        
        elif schema_type == "array":
            items_schema = schema.get("items", {})
            return [self._generate_example_value(items_schema, spec)]
        
        else:
            return "example_value"
    
    def _extract_request_schema(self, spec: Dict[str, Any], endpoint: str) -> Optional[Dict[str, Any]]:
        """Extract request schema for validation"""
        try:
            path_info = spec.get("paths", {}).get(endpoint, {})
            
            for method, operation in path_info.items():
                request_body = operation.get("requestBody", {})
                if request_body:
                    content = request_body.get("content", {})
                    json_content = content.get("application/json", {})
                    schema = json_content.get("schema", {})
                    
                    # Resolve $ref if present
                    if "$ref" in schema:
                        return self._resolve_schema_ref(schema["$ref"], spec)
                    
                    return schema
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting request schema: {e}")
            return None
    
    def _resolve_schema_ref(self, ref: str, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve $ref to actual schema"""
        try:
            ref_path = ref.split("/")
            schema = spec
            for part in ref_path[1:]:  # Skip '#'
                schema = schema.get(part, {})
            return schema
        except Exception as e:
            logger.error(f"Error resolving schema ref {ref}: {e}")
            return {}
    
    def _format_validation_errors(self, errors: List[jsonschema.ValidationError]) -> List[ValidationError]:
        """Format validation errors for response"""
        formatted_errors = []
        
        for error in errors:
            path = ".".join(str(p) for p in error.absolute_path) if error.absolute_path else "root"
            formatted_errors.append(ValidationError(
                path=path,
                msg=error.message
            ))
        
        return formatted_errors
    
    def _create_postman_request(
        self, 
        path: str, 
        method: str, 
        operation: Dict[str, Any], 
        spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create Postman request item"""
        request_item = {
            "name": operation.get("summary", f"{method} {path}"),
            "request": {
                "method": method,
                "header": [
                    {
                        "key": "Content-Type",
                        "value": "application/json"
                    }
                ],
                "url": {
                    "raw": "{{baseUrl}}" + path,
                    "host": ["{{baseUrl}}"],
                    "path": path.strip("/").split("/")
                }
            }
        }
        
        # Add request body if present
        request_body = operation.get("requestBody", {})
        if request_body:
            example_body = self._generate_example_request(spec, path)
            request_item["request"]["body"] = {
                "mode": "raw",
                "raw": json.dumps(example_body, indent=2)
            }
        
        return request_item
    
    def _error_response(self, message: str) -> DevCopilotResponse:
        """Create error response"""
        return DevCopilotResponse(
            endpoint="",
            snippet=CodeSnippet(lang="text", code=f"Error: {message}"),
            example_request={},
            validation=ValidationResult(ok=False, errors=[ValidationError(path="", msg=message)]),
            postman=PostmanInfo(filename="error.json")
        )

# Test cases for DevCopilot scenarios
def test_devcopilot():
    """Test DevCopilot with golden-path scenarios"""
    print("🧪 Testing DevCopilot Golden Paths")
    print("=" * 50)
    
    copilot = DevCopilot()
    
    test_cases = [
        {
            "name": "Invalid payload shows precise pointer fixes",
            "service": "payments",
            "endpoint": "/payments",
            "lang": "python",
            "sample": {"amount": -10, "currency": "EUR"},  # Invalid: negative amount, wrong currency
            "expected_validation_errors": True
        },
        {
            "name": "Valid payload passes validation",
            "service": "payments", 
            "endpoint": "/payments",
            "lang": "curl",
            "sample": {"amount": 100.0, "currency": "USD", "accountId": "acc_123"},
            "expected_validation_errors": False
        },
        {
            "name": "Code snippet generation for JavaScript",
            "service": "payments",
            "endpoint": "/payments",
            "lang": "javascript",
            "sample": None,
            "expected_validation_errors": False
        }
    ]
    
    passed = 0
    total = len(test_cases)
    
    for i, case in enumerate(test_cases, 1):
        try:
            print(f"{i}. {case['name']}")
            
            result = copilot.generate_code_guide(
                service=case["service"],
                endpoint=case["endpoint"],
                lang=case["lang"],
                sample=case["sample"]
            )
            
            # Validate response structure
            valid_structure = (
                hasattr(result, 'endpoint') and
                hasattr(result, 'snippet') and
                hasattr(result, 'example_request') and
                hasattr(result, 'validation') and
                hasattr(result, 'postman')
            )
            
            # Check validation expectations
            validation_ok = (
                (case["expected_validation_errors"] and not result.validation.ok) or
                (not case["expected_validation_errors"] and result.validation.ok)
            )
            
            # Check code snippet is generated
            snippet_ok = len(result.snippet.code) > 10 and case["lang"] in result.snippet.code.lower()
            
            # Check Postman file
            postman_ok = result.postman.filename.endswith(".json")
            
            success = valid_structure and validation_ok and snippet_ok and postman_ok
            status = "✅ PASS" if success else "❌ FAIL"
            
            print(f"   Service: {case['service']}")
            print(f"   Endpoint: {result.endpoint}")
            print(f"   Language: {result.snippet.lang}")
            print(f"   Validation OK: {result.validation.ok}")
            print(f"   Validation Errors: {len(result.validation.errors)}")
            print(f"   Postman File: {result.postman.filename}")
            print(f"   Status: {status}")
            
            if success:
                passed += 1
            else:
                print(f"   Failure reasons:")
                if not valid_structure:
                    print(f"     - Invalid response structure")
                if not validation_ok:
                    print(f"     - Validation expectation not met")
                if not snippet_ok:
                    print(f"     - Code snippet issue")
                if not postman_ok:
                    print(f"     - Postman file issue")
            
            print()
            
        except Exception as e:
            print(f"   ❌ FAIL - Exception: {str(e)}")
            print()
    
    print(f"📊 Results: {passed}/{total} tests passed")
    return passed == total

if __name__ == "__main__":
    test_devcopilot()
