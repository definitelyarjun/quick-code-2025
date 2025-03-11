const API_URL = 'http://localhost:8000';

export const api = {
  async uploadPDF(file) {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_URL}/ingest`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      
      return response.json();
    } catch (error) {
      console.error('Error uploading PDF:', error);
      throw error;
    }
  },

  async generateSchedule(data) {
    try {
      const response = await fetch(`${API_URL}/schedule`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      
      return response.json();
    } catch (error) {
      console.error('Error generating schedule:', error);
      throw error;
    }
  },

  async queryDocuments(query, options = {}) {
    const { method = 'POST', maxTokens = 512, temperature = 0.7, topK = 5 } = options;
    
    try {
      let response;
      
      if (method === 'GET') {
        const params = new URLSearchParams({
          query: query,
          max_tokens: maxTokens,
          temperature: temperature,
          top_k: topK
        });
        
        response = await fetch(`${API_URL}/query?${params.toString()}`);
      } else {
        response = await fetch(`${API_URL}/query`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ 
            query,
            max_tokens: maxTokens,
            temperature: temperature,
            top_k: topK
          }),
        });
      }
      
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      
      const data = await response.json();
      
      // Format the response to match our frontend expectations
      const result = {
        answer: data.answer || data.response || '',
        metadata: data.metadata || {}
      };
      
      // Handle different source formats from the backend
      if (data.sources) {
        result.sources = data.sources;
      } else if (data.relevant_chunks) {
        result.sources = data.relevant_chunks;
      } else if (data.chunks) {
        result.sources = data.chunks;
      } else {
        result.sources = [];
      }
      
      return result;
    } catch (error) {
      console.error('Error querying documents:', error);
      throw error;
    }
  },
  
  async checkModelStatus() {
    try {
      const response = await fetch(`${API_URL}/model-status`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      
      return response.json();
    } catch (error) {
      console.error('Error checking model status:', error);
      throw error;
    }
  }
};
