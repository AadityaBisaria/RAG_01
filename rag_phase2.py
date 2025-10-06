"""
RAG System Phase 2: LLM Integration with LM Studio
Connects retrieval system with LLM for answer generation
"""

import json
import requests
from typing import List, Dict, Any, Optional
import re

class LMStudioClient:
    """Client for LM Studio local API"""
    
    def __init__(self, base_url: str = "http://localhost:1234/v1"):
        """
        Initialize LM Studio client
        
        Args:
            base_url: LM Studio API endpoint (default is localhost:1234)
        """
        self.base_url = base_url
        self.chat_endpoint = f"{base_url}/chat/completions"
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test if LM Studio is running and accessible"""
        try:
            response = requests.get(f"{self.base_url}/models", timeout=5)
            if response.status_code == 200:
                models = response.json()
                print(f"✓ Connected to LM Studio")
                if models.get('data'):
                    model_name = models['data'][0]['id']
                    print(f"  Active model: {model_name}")
                    
                    # Check if it's the expected Hermes model
                    if "hermes" in model_name.lower() or "llama-3.2" in model_name.lower():
                        print(f"  🎯 Detected Hermes/Llama-3.2 model: {model_name}")
                    else:
                        print(f"  ℹ️  Using model: {model_name}")
                else:
                    print("  ⚠️  No model loaded in LM Studio")
            else:
                print("⚠ LM Studio connection issue - check if server is running")
        except requests.exceptions.RequestException as e:
            print(f"⚠ Could not connect to LM Studio at {self.base_url}")
            print(f"  Make sure LM Studio is running with 'Start Server' enabled")
            print(f"  Error: {e}")
    
    def generate(self, prompt: str, temperature: float = 0.1, max_tokens: int = 1000) -> str:
        """
        Generate response from LM Studio
        
        Args:
            prompt: The prompt to send to the model
            temperature: Controls randomness (0.0-1.0, lower = more focused)
            max_tokens: Maximum response length
        
        Returns:
            Generated text response
        """
        payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant that answers questions based on provided documentation."},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        try:
            response = requests.post(self.chat_endpoint, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
        
        except requests.exceptions.RequestException as e:
            return f"Error communicating with LM Studio: {e}"
    
    def generate_stream(self, prompt: str, temperature: float = 0.1, max_tokens: int = 1000):
        """
        Generate streaming response from LM Studio (token by token)
        
        Yields:
            Token strings as they're generated
        """
        payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant that answers questions based on provided documentation."},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }
        
        try:
            response = requests.post(self.chat_endpoint, json=payload, stream=True, timeout=60)
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]  # Remove 'data: ' prefix
                        if data.strip() == '[DONE]':
                            break
                        try:
                            json_data = json.loads(data)
                            if 'choices' in json_data and len(json_data['choices']) > 0:
                                delta = json_data['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue
        
        except requests.exceptions.RequestException as e:
            yield f"\nError: {e}"


class QueryUnderstanding:
    """Handles query analysis, rewriting, and decomposition"""
    
    def __init__(self, llm_client):
        """
        Initialize query understanding
        
        Args:
            llm_client: LMStudioClient instance for query processing
        """
        self.llm = llm_client
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze query to determine type and complexity
        
        Args:
            query: User's question
            
        Returns:
            Dict with query analysis
        """
        analysis_prompt = f"""Analyze the following query and provide structured information:

Query: "{query}"

Please respond with a JSON object containing:
{{
    "query_type": "simple|multi_hop|clarification|ambiguous|comparison",
    "complexity": "low|medium|high",
    "requires_decomposition": true/false,
    "key_concepts": ["concept1", "concept2"],
    "clarification_needed": true/false,
    "suggested_rewrites": ["rewrite1", "rewrite2"]
}}

Analysis:"""

        try:
            response = self.llm.generate(analysis_prompt, temperature=0.1, max_tokens=500)
            
            # Extract JSON from response
            import json
            import re
            
            # Find JSON in response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
            else:
                # Fallback analysis
                analysis = {
                    "query_type": "simple",
                    "complexity": "low",
                    "requires_decomposition": False,
                    "key_concepts": [],
                    "clarification_needed": False,
                    "suggested_rewrites": []
                }
            
            return analysis
            
        except Exception as e:
            print(f"Warning: Query analysis failed: {e}")
            return {
                "query_type": "simple",
                "complexity": "low",
                "requires_decomposition": False,
                "key_concepts": [],
                "clarification_needed": False,
                "suggested_rewrites": []
            }
    
    def rewrite_query(self, query: str, context: str = "") -> str:
        """
        Rewrite ambiguous or unclear queries
        
        Args:
            query: Original query
            context: Optional context for better rewriting
            
        Returns:
            Rewritten query
        """
        rewrite_prompt = f"""Rewrite the following query to be more specific and clear for document retrieval:

Original Query: "{query}"
Context: {context if context else "No additional context"}

Instructions:
1. Make the query more specific if it's ambiguous
2. Use clear, searchable terms
3. Keep the original intent
4. Don't change factual questions

Rewritten Query:"""

        try:
            rewritten = self.llm.generate(rewrite_prompt, temperature=0.1, max_tokens=200)
            return rewritten.strip().strip('"')
        except Exception as e:
            print(f"Warning: Query rewriting failed: {e}")
            return query
    
    def decompose_query(self, query: str) -> List[str]:
        """
        Decompose complex queries into simpler sub-queries
        
        Args:
            query: Complex query to decompose
            
        Returns:
            List of simpler sub-queries
        """
        decomposition_prompt = f"""Break down this complex query into simpler, answerable sub-queries:

Complex Query: "{query}"

Instructions:
1. Identify the main components of the question
2. Create 2-4 simpler sub-queries that can be answered independently
3. Each sub-query should be clear and specific
4. Maintain the logical flow of the original question

Respond with a JSON array of sub-queries:
["sub-query 1", "sub-query 2", "sub-query 3"]

Sub-queries:"""

        try:
            response = self.llm.generate(decomposition_prompt, temperature=0.1, max_tokens=400)
            
            import json
            import re
            
            # Find JSON array in response
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                sub_queries = json.loads(json_match.group())
                return [q.strip().strip('"') for q in sub_queries if q.strip()]
            else:
                return [query]  # Fallback to original
                
        except Exception as e:
            print(f"Warning: Query decomposition failed: {e}")
            return [query]


class ChunkCompressor:
    """Compresses chunks to reduce context window usage"""
    
    def __init__(self, llm_client):
        """
        Initialize chunk compressor
        
        Args:
            llm_client: LMStudioClient instance for summarization
        """
        self.llm = llm_client
    
    def compress_chunks(self, chunks: List[Dict[str, Any]], max_tokens_per_chunk: int = 200) -> List[Dict[str, Any]]:
        """
        Compress chunks by summarizing them
        
        Args:
            chunks: List of chunk dictionaries
            max_tokens_per_chunk: Target token count per compressed chunk
            
        Returns:
            List of compressed chunks
        """
        compressed_chunks = []
        
        for chunk in chunks:
            content = chunk.get('content', '')
            metadata = chunk.get('metadata', {})
            
            # If chunk is already short enough, keep as is
            if len(content.split()) <= max_tokens_per_chunk * 0.75:  # Rough token estimation
                compressed_chunks.append(chunk)
                continue
            
            # Summarize the chunk
            summary_prompt = f"""Summarize the following text in {max_tokens_per_chunk} tokens or less, preserving key information:

{content}

Summary:"""
            
            try:
                summary = self.llm.generate(summary_prompt, temperature=0.1, max_tokens=max_tokens_per_chunk)
                
                compressed_chunk = {
                    'content': summary.strip(),
                    'metadata': {
                        **metadata,
                        'compressed': True,
                        'original_length': len(content),
                        'compressed_length': len(summary)
                    },
                    'id': f"{chunk.get('id', '')}_compressed"
                }
                compressed_chunks.append(compressed_chunk)
                
            except Exception as e:
                print(f"Warning: Failed to compress chunk: {e}")
                # Fall back to truncation
                words = content.split()
                truncated = ' '.join(words[:max_tokens_per_chunk])
                compressed_chunk = {
                    'content': truncated + "...",
                    'metadata': {
                        **metadata,
                        'compressed': True,
                        'method': 'truncated',
                        'original_length': len(content)
                    },
                    'id': f"{chunk.get('id', '')}_truncated"
                }
                compressed_chunks.append(compressed_chunk)
        
        return compressed_chunks


class CitationVerifier:
    """Verifies and corrects citations in LLM responses"""
    
    def __init__(self):
        """Initialize citation verifier"""
        pass
    
    def extract_citations_from_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Extract citations from LLM response
        
        Args:
            response: LLM response text
            
        Returns:
            List of citation dictionaries
        """
        import re
        
        citations = []
        
        # Pattern for [Source: filename] citations
        source_pattern = r'\[Source:\s*([^\]]+)\]'
        source_matches = re.finditer(source_pattern, response)
        
        for match in source_matches:
            citations.append({
                'type': 'source',
                'text': match.group(0),
                'source': match.group(1).strip(),
                'start': match.start(),
                'end': match.end()
            })
        
        # Pattern for other citation formats
        other_patterns = [
            r'\[([^\]]+)\]',  # [anything in brackets]
            r'\(Source:\s*([^)]+)\)',  # (Source: filename)
            r'According to ([^,.]*),',  # According to filename,
        ]
        
        for pattern in other_patterns:
            matches = re.finditer(pattern, response)
            for match in matches:
                if not any(c['start'] <= match.start() < c['end'] for c in citations):
                    citations.append({
                        'type': 'other',
                        'text': match.group(0),
                        'content': match.group(1) if match.groups() else match.group(0),
                        'start': match.start(),
                        'end': match.end()
                    })
        
        return citations
    
    def verify_citations_against_chunks(self, response: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Verify citations against retrieved chunks
        
        Args:
            response: LLM response
            retrieved_chunks: List of retrieved chunks with metadata
            
        Returns:
            Dict with verification results
        """
        # Extract citations from response
        response_citations = self.extract_citations_from_response(response)
        
        # Get available sources from chunks
        available_sources = set()
        for chunk in retrieved_chunks:
            metadata = chunk.get('metadata', {})
            source = metadata.get('source', 'Unknown')
            available_sources.add(source)
        
        # Verify citations
        verified_citations = []
        missing_citations = []
        
        for citation in response_citations:
            if citation['type'] == 'source':
                source = citation['source']

                # Fallback mapping: convert patterns like "Document N" to the actual filename
                try:
                    import re as _re
                    doc_match = _re.match(r'(?i)document\s+(\d+)', source.strip())
                except Exception:
                    doc_match = None
                if doc_match:
                    idx = int(doc_match.group(1)) - 1
                    if 0 <= idx < len(retrieved_chunks):
                        mapped_source = retrieved_chunks[idx].get('metadata', {}).get('source', source)
                        citation = {
                            **citation,
                            'source': mapped_source,
                            'text': f"[Source: {mapped_source}]"
                        }

                if citation['source'] in available_sources:
                    verified_citations.append(citation)
                else:
                    missing_citations.append(citation)
        
        # Find uncited chunks (chunks that should be cited but aren't)
        response_lower = response.lower()
        uncited_chunks = []
        
        for chunk in retrieved_chunks:
            metadata = chunk.get('metadata', {})
            source = metadata.get('source', 'Unknown')
            
            # Check if this source is mentioned in the response
            if source.lower() not in response_lower:
                # Check if chunk content is referenced
                content_words = chunk.get('content', '').lower().split()[:10]  # First 10 words
                content_mentioned = any(word in response_lower for word in content_words if len(word) > 3)
                
                if content_mentioned:
                    uncited_chunks.append({
                        'source': source,
                        'content_preview': chunk.get('content', '')[:100] + '...'
                    })
        
        # Coverage based on unique sources to avoid >100% anomalies
        unique_verified_sources = {c['source'] for c in verified_citations}
        unique_retrieved_sources = {c.get('metadata', {}).get('source', 'Unknown') for c in retrieved_chunks}
        coverage = len(unique_verified_sources) / max(len(unique_retrieved_sources), 1)

        return {
            'verified_citations': verified_citations,
            'missing_citations': missing_citations,
            'uncited_chunks': uncited_chunks,
            'citation_coverage': coverage
        }
    
    def generate_citation_report(self, verification_result: Dict[str, Any]) -> str:
        """
        Generate a human-readable citation report
        
        Args:
            verification_result: Result from verify_citations_against_chunks
            
        Returns:
            Formatted report string
        """
        report = []
        
        verified = verification_result['verified_citations']
        missing = verification_result['missing_citations']
        uncited = verification_result['uncited_chunks']
        coverage = verification_result['citation_coverage']
        
        report.append(f"📊 Citation Report (Coverage: {coverage:.1%})")
        report.append("=" * 50)
        
        if verified:
            report.append(f"✅ Verified Citations ({len(verified)}):")
            for cite in verified:
                report.append(f"  - {cite['text']}")
        
        if missing:
            report.append(f"\n⚠️  Missing/Invalid Citations ({len(missing)}):")
            for cite in missing:
                report.append(f"  - {cite['text']}")
        
        if uncited:
            report.append(f"\n📝 Potentially Uncited Content ({len(uncited)}):")
            for chunk in uncited:
                report.append(f"  - {chunk['source']}: {chunk['content_preview']}")
        
        return "\n".join(report)


class PromptBuilder:
    """Builds optimized prompts for RAG system"""
    
    @staticmethod
    def build_rag_prompt(query: str, retrieved_chunks: List[Dict[str, Any]], structured_output: bool = False) -> str:
        """
        Build RAG prompt with context and instructions
        
        Args:
            query: User's question
            retrieved_chunks: List of relevant document chunks from ChromaDB
            structured_output: Whether to request structured JSON output
        
        Returns:
            Formatted prompt for LLM
        """
        # Format context from retrieved chunks
        context_sections = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            content = chunk['documents'][0] if isinstance(chunk.get('documents'), list) else chunk.get('content', '')
            metadata = chunk['metadatas'][0] if isinstance(chunk.get('metadatas'), list) else chunk.get('metadata', {})
            source = metadata.get('source', 'Unknown')
            
            context_sections.append(f"[Document {i}] [Source: {source}]\n{content}\n")
        
        context = "\n".join(context_sections)
        
        if structured_output:
            # Build structured prompt
            prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided documentation.

CONTEXT FROM DOCUMENTATION:
{context}

INSTRUCTIONS:
1. Answer the question using ONLY information from the context above
2. If the answer is in the context, provide a clear and concise response
3. For EVERY sentence that uses document information, append a citation using the exact filename shown in context, like [Source: filename]. Do NOT use "Document N". If a sentence uses multiple sources, cite them like [Source: fileA; fileB].
4. If the information is NOT in the provided context, respond with: "I couldn't find this information in the provided documents."
5. Do not make up information or use external knowledge
6. Be specific and reference the relevant document when possible

QUESTION: {query}

Please respond with a JSON object in this exact format:
{{
    "answer": "Your answer here",
    "citations": ["source1", "source2"],
    "confidence": "high|medium|low",
    "information_found": true/false
}}

RESPONSE:"""
        else:
            # Build regular prompt
            prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided documentation.

CONTEXT FROM DOCUMENTATION:
{context}

INSTRUCTIONS:
1. Answer the question using ONLY information from the context above
2. If the answer is in the context, provide a clear and concise response
3. For EVERY sentence that uses document information, append a citation using the exact filename shown in context, like [Source: filename]. Do NOT use "Document N". If a sentence uses multiple sources, cite them like [Source: fileA; fileB].
4. If the information is NOT in the provided context, respond with: "I couldn't find this information in the provided documents."
5. Do not make up information or use external knowledge
6. Be specific and reference the relevant document when possible

QUESTION: {query}

ANSWER:"""
        
        return prompt
    
    @staticmethod
    def extract_citations(response: str) -> tuple[str, List[str]]:
        """
        Extract citations from LLM response
        
        Returns:
            Tuple of (cleaned_response, list_of_citations)
        """
        # Find all citations like [Source: filename]
        citation_pattern = r'\[Source:\s*([^\]]+)\]'
        citations = re.findall(citation_pattern, response)
        
        # Remove duplicate citations while preserving order
        unique_citations = []
        for cite in citations:
            if cite not in unique_citations:
                unique_citations.append(cite)
        
        return response, unique_citations
    
    @staticmethod
    def parse_structured_response(response: str) -> Dict[str, Any]:
        """
        Parse structured JSON response from LLM
        
        Args:
            response: LLM response text
            
        Returns:
            Dict with parsed response or fallback
        """
        import json
        import re
        
        try:
            # Find JSON in response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                structured_data = json.loads(json_match.group())
                return structured_data
        except json.JSONDecodeError:
            pass
        
        # Fallback to regular parsing
        return {
            'answer': response,
            'citations': PromptBuilder.extract_citations(response)[1],
            'confidence': 'unknown',
            'information_found': True
        }


class RAGSystem:
    """Complete RAG system with retrieval and generation"""
    
    def __init__(self, chroma_db_manager, llm_client: LMStudioClient):
        """
        Initialize RAG system
        
        Args:
            chroma_db_manager: ChromaDBManager instance from Phase 1
            llm_client: LMStudioClient instance
        """
        self.db = chroma_db_manager
        self.llm = llm_client
        self.prompt_builder = PromptBuilder()
        self.compressor = ChunkCompressor(llm_client)
        self.query_understanding = QueryUnderstanding(llm_client)
        self.citation_verifier = CitationVerifier()
    
    def query(self, question: str, n_results: int = 5, stream: bool = False, use_reranking: bool = True, retrieve_n: int = 10, compress_chunks: bool = False, max_tokens_per_chunk: int = 200, structured_output: bool = False, analyze_query: bool = True, verify_citations: bool = True, preview_chars: int = 300) -> Dict[str, Any]:
        """
        Main RAG query pipeline with optional reranking
        
        Args:
            question: User's question
            n_results: Number of chunks to return after reranking
            stream: Whether to stream the response
            use_reranking: Whether to use reranking for better results
            retrieve_n: Number of chunks to retrieve initially (only used with reranking)
            compress_chunks: Whether to compress chunks before sending to LLM
            max_tokens_per_chunk: Maximum tokens per chunk when compressing
            structured_output: Whether to use structured JSON output
            analyze_query: Whether to analyze and potentially rewrite the query
            verify_citations: Whether to verify citations after generation
        
        Returns:
            Dict with answer, citations, retrieved chunks, and analysis
        """
        print(f"\n{'='*60}")
        print(f"Query: {question}")
        print(f"{'='*60}\n")
        
        # Step 0: Analyze query (optional)
        query_analysis = None
        final_question = question
        if analyze_query:
            print("🧠 Step 0: Analyzing query...")
            query_analysis = self.query_understanding.analyze_query(question)
            
            print(f"  Query type: {query_analysis.get('query_type', 'unknown')}")
            print(f"  Complexity: {query_analysis.get('complexity', 'unknown')}")
            
            # Rewrite if needed
            if query_analysis.get('clarification_needed', False):
                print("  Rewriting query for clarity...")
                final_question = self.query_understanding.rewrite_query(question)
                print(f"  Rewritten: {final_question}")
            
            # Decompose if complex
            sub_queries = []
            if query_analysis.get('requires_decomposition', False):
                print("  Decomposing complex query...")
                sub_queries = self.query_understanding.decompose_query(final_question)
                print(f"  Sub-queries: {sub_queries}")
            
            print()
        
        # Step 1: Retrieve relevant chunks (with optional reranking)
        print("🔍 Step 1: Retrieving relevant documents...")
        if use_reranking:
            print(f"  Using reranking: retrieve {retrieve_n} chunks, return top {n_results}")
            results = self.db.query_with_reranking(final_question, retrieve_n=retrieve_n, final_n=n_results)
        else:
            results = self.db.query(final_question, n_results=n_results)
        
        if not results['documents'][0]:
            return {
                'answer': "I couldn't find any relevant information in the provided documents.",
                'citations': [],
                'retrieved_chunks': []
            }
        
        # Show what was retrieved
        print(f"✓ Retrieved {len(results['documents'][0])} relevant chunks\n")
        for i, metadata in enumerate(results['metadatas'][0][:3], 1):
            print(f"  {i}. {metadata.get('source', 'Unknown')}")
        
        # Step 2: Compress chunks if requested
        print("\n📝 Step 2: Processing chunks...")
        retrieved_chunks = []
        for doc, meta, dist, _id in zip(results['documents'][0], results['metadatas'][0], results['distances'][0], results['ids'][0]):
            retrieved_chunks.append({
                'content': doc,
                'metadata': meta,
                'similarity': 1 - dist,
                'id': _id
            })
        
        if compress_chunks:
            print("  Compressing chunks to reduce context usage...")
            retrieved_chunks = self.compressor.compress_chunks(retrieved_chunks, max_tokens_per_chunk)
            print(f"✓ Compressed {len(retrieved_chunks)} chunks")
        
        # Print the exact chunks that will be sent to the LLM
        print("📄 Top chunks sent to LLM:")
        for idx, ch in enumerate(retrieved_chunks, 1):
            src = ch.get('metadata', {}).get('source', 'Unknown')
            cid = ch.get('id', 'N/A')
            sim = ch.get('similarity', 0.0)
            text = ch.get('content', '')
            preview = (text[:preview_chars] + ("..." if len(text) > preview_chars else ""))
            print(f"  [{idx}] source={src} | id={cid} | similarity={sim:.3f}")
            print(f"      {preview}")

        prompt = self.prompt_builder.build_rag_prompt(final_question, retrieved_chunks, structured_output)
        print("✓ Prompt ready\n")
        
        # Step 3: Generate answer
        print("🤖 Step 3: Generating answer with LLM...\n")
        
        if stream:
            print("Answer: ", end='', flush=True)
            full_response = ""
            for token in self.llm.generate_stream(prompt):
                print(token, end='', flush=True)
                full_response += token
            print("\n")
            answer = full_response
        else:
            answer = self.llm.generate(prompt)
            print(f"Answer: {answer}\n")
        
        # Step 4: Process response and extract citations
        if structured_output:
            print("\n📋 Step 4: Parsing structured response...")
            structured_response = self.prompt_builder.parse_structured_response(answer)
            answer = structured_response.get('answer', answer)
            citations = structured_response.get('citations', [])
            confidence = structured_response.get('confidence', 'unknown')
            print(f"✓ Confidence: {confidence}")
        else:
            _, citations = self.prompt_builder.extract_citations(answer)
        
        # Step 5: Verify citations (optional)
        citation_report = None
        if verify_citations:
            print("\n🔍 Step 5: Verifying citations...")
            verification_result = self.citation_verifier.verify_citations_against_chunks(answer, retrieved_chunks)
            citation_report = self.citation_verifier.generate_citation_report(verification_result)
            print(f"✓ Citation coverage: {verification_result['citation_coverage']:.1%}")
        
        print(f"{'='*60}\n")
        
        result = {
            'answer': answer,
            'citations': citations,
            'retrieved_chunks': retrieved_chunks,
            'query': question,
            'final_question': final_question
        }
        
        if query_analysis:
            result['query_analysis'] = query_analysis
        
        if structured_output:
            result['structured_response'] = structured_response
        
        if citation_report:
            result['citation_report'] = citation_report
        
        return result
    
    def batch_test(self, test_queries: List[str], n_results: int = 5, use_reranking: bool = True, compress_chunks: bool = False, structured_output: bool = False, analyze_query: bool = True):
        """
        Test RAG system with multiple queries
        
        Args:
            test_queries: List of test questions
            n_results: Number of chunks to retrieve per query
            use_reranking: Whether to use reranking
            compress_chunks: Whether to compress chunks
            structured_output: Whether to use structured output
            analyze_query: Whether to analyze queries
        """
        results = []
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{'#'*60}")
            print(f"TEST {i}/{len(test_queries)}")
            print(f"{'#'*60}")
            
            result = self.query(query, n_results=n_results, use_reranking=use_reranking, compress_chunks=compress_chunks, structured_output=structured_output, analyze_query=analyze_query)
            results.append(result)
            
            # Show citations
            if result['citations']:
                print(f"📚 Citations: {', '.join(result['citations'])}")
            else:
                print("📚 No citations found in response")
        
        return results
    
    def interactive_mode(self):
        """Run RAG system in interactive mode"""
        print(f"\n{'='*60}")
        print("RAG System - Interactive Mode")
        print("Commands: 'quit' to exit, 'stream' to toggle streaming, 'rerank' to toggle reranking, 'compress' to toggle compression")
        print("          'analyze' to toggle query analysis, 'structured' to toggle structured output, 'verify' to toggle citation verification")
        print(f"{'='*60}\n")
        
        stream_mode = False
        rerank_mode = True
        compress_mode = False
        analyze_mode = True
        structured_mode = False
        verify_mode = True
        
        while True:
            try:
                user_input = input("\n💬 Your question: ").strip()
                
                if user_input.lower() == 'quit':
                    print("Goodbye! 👋")
                    break
                
                if user_input.lower() == 'stream':
                    stream_mode = not stream_mode
                    print(f"Streaming mode: {'ON' if stream_mode else 'OFF'}")
                    continue
                
                if user_input.lower() == 'rerank':
                    rerank_mode = not rerank_mode
                    print(f"Reranking mode: {'ON' if rerank_mode else 'OFF'}")
                    continue
                
                if user_input.lower() == 'compress':
                    compress_mode = not compress_mode
                    print(f"Compression mode: {'ON' if compress_mode else 'OFF'}")
                    continue
                
                if user_input.lower() == 'analyze':
                    analyze_mode = not analyze_mode
                    print(f"Query analysis mode: {'ON' if analyze_mode else 'OFF'}")
                    continue
                
                if user_input.lower() == 'structured':
                    structured_mode = not structured_mode
                    print(f"Structured output mode: {'ON' if structured_mode else 'OFF'}")
                    continue
                
                if user_input.lower() == 'verify':
                    verify_mode = not verify_mode
                    print(f"Citation verification mode: {'ON' if verify_mode else 'OFF'}")
                    continue
                
                if not user_input:
                    continue
                
                self.query(user_input, stream=stream_mode, use_reranking=rerank_mode, compress_chunks=compress_mode, structured_output=structured_mode, analyze_query=analyze_mode, verify_citations=verify_mode)
            
            except KeyboardInterrupt:
                print("\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"Error: {e}")


# Example usage
if __name__ == "__main__":
    # Import Phase 1 components
    from rag_phase1 import ChromaDBManager, RAGDataPipeline
    
    print("\n" + "="*60)
    print("RAG SYSTEM PHASE 2: LLM Integration")
    print("="*60 + "\n")
    
    # Initialize components
    print("Initializing RAG system components...\n")
    
    # Initialize ChromaDB (assumes Phase 1 was run)
    db_manager = ChromaDBManager(persist_directory="./chroma_db")
    
    # Check if we have documents
    if db_manager.collection.count() == 0:
        print("⚠ No documents in ChromaDB. Running Phase 1 data ingestion first...\n")
        pipeline = RAGDataPipeline()
        pipeline.ingest_documents("./sample_docs")
    
    # Initialize LM Studio client
    llm_client = LMStudioClient()
    
    # Create RAG system
    rag = RAGSystem(db_manager, llm_client)
    
    # Test queries
    print("\n" + "="*60)
    print("RUNNING TEST QUERIES")
    print("="*60)
    
    test_queries = [
        "How do I install the product?",
        "What are the system requirements?",
        "How can I reset my password?",
        "What payment methods do you accept?",
        "Can I use this on Linux?"  # Should return "not found"
    ]
    
    # Test with query analysis and reranking
    print("\n🧪 Testing with QUERY ANALYSIS and RERANKING:")
    rag.batch_test(test_queries[:3], n_results=3, use_reranking=True, analyze_query=True)
    
    # Test with structured output
    print("\n🧪 Testing with STRUCTURED OUTPUT:")
    rag.batch_test(test_queries[:2], n_results=3, structured_output=True)
    
    # Test with compression
    print("\n🧪 Testing with COMPRESSION:")
    rag.batch_test(test_queries[:2], n_results=3, use_reranking=True, compress_chunks=True)
    
    # Test complex query decomposition
    print("\n🧪 Testing COMPLEX QUERY DECOMPOSITION:")
    complex_queries = [
        "What are the installation steps and system requirements for the product?",
        "How do I reset my password and what payment methods are available?"
    ]
    rag.batch_test(complex_queries, n_results=5, analyze_query=True, structured_output=True)
    
    # Optional: Start interactive mode
    print("\n" + "="*60)
    print("Starting interactive mode...")
    print("="*60)
    
    rag.interactive_mode()