const API_URL = 'http://localhost:8000';

export const api = {
  async uploadPDF(file) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_URL}/ingest`, {
      method: 'POST',
      body: formData,
    });
    return response.json();
  },

  async generateSchedule(data) {
    const response = await fetch(`${API_URL}/schedule`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });
    return response.json();
  },

  async queryDocuments(query, method = 'POST') {
    if (method === 'GET') {
      const response = await fetch(`${API_URL}/query?query=${encodeURIComponent(query)}`);
      return response.json();
    } else {
      const response = await fetch(`${API_URL}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query }),
      });
      return response.json();
    }
  },
};
