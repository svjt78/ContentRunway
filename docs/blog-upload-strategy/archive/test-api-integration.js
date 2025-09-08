#!/usr/bin/env node

import fetch from 'node-fetch';

console.log('Testing API Integration - Programmatic Upload');

// Replace with your actual API token
const API_TOKEN = process.env.API_TOKEN || 'api_xxxxxxxxxxxxxxxxxxxxxxxx';
const BASE_URL = 'http://localhost:3003';

class DigitalDossierAPI {
  constructor(apiToken, baseUrl = BASE_URL) {
    this.apiToken = apiToken;
    this.baseUrl = baseUrl;
  }

  async getGenres() {
    const response = await fetch(`${this.baseUrl}/api/genres`, {
      headers: { 'Authorization': `Bearer ${this.apiToken}` }
    });
    return response.json();
  }

  async uploadSingleDocument(document) {
    const response = await fetch(`${this.baseUrl}/api/upload/programmatic`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.apiToken}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(document)
    });
    return response.json();
  }

  async uploadBatchDocuments(documents) {
    const response = await fetch(`${this.baseUrl}/api/upload/programmatic/batch`, {
      method: 'POST', 
      headers: {
        'Authorization': `Bearer ${this.apiToken}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ documents })
    });
    return response.json();
  }
}

const testDocument = {
  title: 'Test API Integration Document',
  author: 'API Test Suite',
  category: 'Book',
  genreId: 1,
  summary: 'Testing programmatic upload functionality',
  content: 'This is a test document created via the programmatic API to validate the integration.',
  coverImage: {
    data: 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==',
    filename: 'test-cover.png',
    mimeType: 'image/png'
  },
  pdfFile: {
    data: 'JVBERi0xLjQKJdPr6eEKMSAwIG9iago8PAovVHlwZSAvQ2F0YWxvZwovUGFnZXMgMiAwIFIKPj4KZW5kb2JqCjIgMCBvYmoKPDwKL1R5cGUgL1BhZ2VzCi9LaWRzIFszIDAgUl0KL0NvdW50IDEKPD4KZW5kb2JqCjMgMCBvYmoKPDwKL1R5cGUgL1BhZ2UKL1BhcmVudCAyIDAgUgovTWVkaWFCb3ggWzAgMCA2MTIgNzkyXQovUmVzb3VyY2VzIDw8Ci9Gb250IDw8Ci9GMSA0IDAgUgo+Pgo+PgovQ29udGVudHMgNSAwIFIKPj4KZW5kb2JqCjQgMCBvYmoKPDwKL1R5cGUgL0ZvbnQKL1N1YnR5cGUgL1R5cGUxCi9CYXNlRm9udCAvSGVsdmV0aWNhCj4+CmVuZG9iago1IDAgb2JqCjw8Ci9MZW5ndGggNDQKPj4Kc3RyZWFtCkJUCi9GMSA5OSBUZgo1MCA3NTAgVGQKKFRlc3QgUERGIERvY3VtZW50KSBUagpFVApzdHJlYW0KZW5kb2JqCnhyZWYKMCA2CjAwMDAwMDAwMDAgNjU1MzUgZiAKMDAwMDAwMDAwOSAwMDAwMCBuIAowMDAwMDAwMDc0IDAwMDAwIG4gCjAwMDAwMDAxMjAgMDAwMDAgbiAKMDAwMDAwMDI3MSAwMDAwMCBuIAowMDAwMDAwMzE4IDAwMDAwIG4gCnRyYWlsZXIKPDwKL1NpemUgNgovUm9vdCAxIDAgUgo+PgpzdGFydHhyZWYKNDEwCiUlRU9G',
    filename: 'test-document.pdf',
    mimeType: 'application/pdf'
  }
};

const batchTestDocuments = [
  {
    title: 'Batch Test Document 1',
    author: 'Batch Tester',
    category: 'Blog',
    genreId: 1,
    content: 'First batch test document',
    coverImage: {
      data: 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==',
      filename: 'batch-test-1.png',
      mimeType: 'image/png'
    }
  },
  {
    title: 'Batch Test Document 2',
    author: 'Batch Tester',
    category: 'Book',
    genreId: 2,
    summary: 'Second batch test document',
    coverImage: {
      data: 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==',
      filename: 'batch-test-2.png',
      mimeType: 'image/png'
    },
    pdfFile: {
      data: 'JVBERi0xLjQKJdPr6eEKMSAwIG9iago8PAovVHlwZSAvQ2F0YWxvZwovUGFnZXMgMiAwIFIKPj4KZW5kb2JqCjIgMCBvYmoKPDwKL1R5cGUgL1BhZ2VzCi9LaWRzIFszIDAgUl0KL0NvdW50IDEKPD4KZW5kb2JqCjMgMCBvYmoKPDwKL1R5cGUgL1BhZ2UKL1BhcmVudCAyIDAgUgovTWVkaWFCb3ggWzAgMCA2MTIgNzkyXQovUmVzb3VyY2VzIDw8Ci9Gb250IDw8Ci9GMSA0IDAgUgo+Pgo+PgovQ29udGVudHMgNSAwIFIKPj4KZW5kb2JqCjQgMCBvYmoKPDwKL1R5cGUgL0ZvbnQKL1N1YnR5cGUgL1R5cGUxCi9CYXNlRm9udCAvSGVsdmV0aWNhCj4+CmVuZG9iago1IDAgb2JqCjw8Ci9MZW5ndGggNTUKPj4Kc3RyZWFtCkJUCi9GMSA5OSBUZgo1MCA3NTAgVGQKKEJhdGNoIFRlc3QgUERGIERvY3VtZW50IDIpIFRqCkVUCnN0cmVhbQplbmRvYmoKeHJlZgowIDYKMDAwMDAwMDAwMCA2NTUzNSBmIAowMDAwMDAwMDA5IDAwMDAwIG4gCjAwMDAwMDAwNzQgMDAwMDAgbiAKMDAwMDAwMDEyMCAwMDAwMCBuIAowMDAwMDAwMjcxIDAwMDAwIG4gCjAwMDAwMDAzMTggMDAwMDAgbiAKdHJhaWxlcgo8PAovU2l6ZSA2Ci9Sb290IDEgMCBSCj4+CnN0YXJ0eHJlZgo0MjEKJSVFT0Y=',
      filename: 'batch-test-2.pdf',
      mimeType: 'application/pdf'
    }
  }
];

async function runTests() {
  const api = new DigitalDossierAPI(API_TOKEN);

  try {
    console.log('🔍 Fetching available genres...');
    const genres = await api.getGenres();
    console.log('✅ Available genres:', genres);

    console.log('\n📤 Testing single document upload...');
    const singleResult = await api.uploadSingleDocument(testDocument);
    if (singleResult.success) {
      console.log('✅ Single upload successful:', singleResult.data.title);
      console.log(`📝 Access URL: ${BASE_URL}/${singleResult.data.category.toLowerCase()}s/${singleResult.data.slug}`);
    } else {
      console.error('❌ Single upload failed:', singleResult.error);
    }

    console.log('\n📦 Testing batch document upload...');
    const batchResult = await api.uploadBatchDocuments(batchTestDocuments);
    if (batchResult.success) {
      console.log('✅ Batch upload completed');
      console.log(`📊 Summary: ${batchResult.summary.successful}/${batchResult.summary.total} successful`);
      batchResult.results.forEach((result, index) => {
        if (result.success) {
          console.log(`  ✅ Document ${index + 1}: ${result.data.title}`);
        } else {
          console.log(`  ❌ Document ${index + 1}: ${result.error.message}`);
        }
      });
    } else {
      console.error('❌ Batch upload failed:', batchResult.error);
    }

  } catch (error) {
    console.error('❌ Test failed with error:', error.message);
  }
}

if (API_TOKEN === 'api_xxxxxxxxxxxxxxxxxxxxxxxx') {
  console.log('⚠️  Please set your API token:');
  console.log('   export API_TOKEN=your_actual_api_token');
  console.log('   node test-api-integration.js');
} else {
  runTests();
}