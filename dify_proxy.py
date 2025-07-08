from http.server import BaseHTTPRequestHandler, HTTPServer
from http.client import HTTPConnection
import json

class NativeProxyHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        # Step 1: Read incoming request from Dify
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        incoming = json.loads(body)
        
        print(f"Incoming Dify request: {incoming}")

        # Step 2: Convert Dify schema to reranker schema
        rerank_payload = {
            "query": incoming.get("query", ""),
            "texts": incoming.get("documents", []),  # Dify uses "documents"
            "truncate": True,
            "truncation_direction": "Right",
            "raw_scores": False
        }
        
        print(f"Sending to HF rerank: {rerank_payload}")

        # Step 3: Send request to actual rerank server at localhost:8085
        conn = HTTPConnection("localhost", 8085)
        rerank_body = json.dumps(rerank_payload)
        headers = {"Content-Type": "application/json"}

        try:
            conn.request("POST", "/rerank", rerank_body, headers)
            resp = conn.getresponse()
            result_data = resp.read()
            conn.close()
            
            print(f"HF rerank response status: {resp.status}")
            print(f"HF rerank response: {result_data.decode()}")

            if resp.status == 200:
                # Step 4: Convert HF response to Dify expected format
                hf_result = json.loads(result_data)
                
                # Convert HF format to Dify format
                dify_response = self.convert_hf_to_dify_format(hf_result, incoming.get("documents", []))
                
                print(f"Converted to Dify format: {dify_response}")
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(dify_response).encode('utf-8'))
            else:
                # Forward error response
                self.send_response(resp.status)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(result_data)

        except Exception as e:
            print(f"Proxy error: {e}")
            # On error, return HTTP 500
            error_response = {"error": f"Proxy error: {e}"}
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(error_response).encode('utf-8'))

    def convert_hf_to_dify_format(self, hf_result, original_documents):
        """
        Convert Hugging Face rerank response to Dify expected format
        
        HF typically returns something like:
        [
          {"corpus_id": 0, "score": 0.85},
          {"corpus_id": 1, "score": 0.72}
        ]
        
        Dify expects:
        {
          "results": [
            {
              "index": 0,
              "document": {"text": "document text"},
              "relevance_score": 0.85
            }
          ]
        }
        """
        results = []
        
        # Handle different possible HF response formats
        if isinstance(hf_result, list):
            # Format 1: Direct list of results
            for item in hf_result:
                if isinstance(item, dict):
                    index = item.get("corpus_id", item.get("index", 0))
                    score = item.get("score", item.get("relevance_score", 0.0))
                    
                    result = {
                        "index": index,
                        "document": {
                            "text": original_documents[index] if index < len(original_documents) else ""
                        },
                        "relevance_score": float(score)
                    }
                    results.append(result)
        
        elif isinstance(hf_result, dict):
            # Format 2: Dict with results key
            if "results" in hf_result:
                for item in hf_result["results"]:
                    index = item.get("corpus_id", item.get("index", 0))
                    score = item.get("score", item.get("relevance_score", 0.0))
                    
                    result = {
                        "index": index,
                        "document": {
                            "text": original_documents[index] if index < len(original_documents) else ""
                        },
                        "relevance_score": float(score)
                    }
                    results.append(result)
            
            # Format 3: Dict with scores key
            elif "scores" in hf_result:
                for i, score in enumerate(hf_result["scores"]):
                    result = {
                        "index": i,
                        "document": {
                            "text": original_documents[i] if i < len(original_documents) else ""
                        },
                        "relevance_score": float(score)
                    }
                    results.append(result)
        
        # Sort by relevance score (highest first)
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return {"results": results}

    def log_message(self, format, *args):
        # Enable logging for debugging
        print(f"[{self.address_string()}] {format % args}")

if __name__ == "__main__":
    print("Native rerank proxy running on http://0.0.0.0:8086/rerank")
    print("This proxy converts between Dify format and Hugging Face rerank format")
    HTTPServer(('0.0.0.0', 8086), NativeProxyHandler).serve_forever()
