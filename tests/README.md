# ContentRunway Test Suite

## 🧪 Test Scripts Usage Guide

This directory contains comprehensive test scripts for debugging and validating the ContentRunway multi-agent pipeline.

## 📁 Directory Structure

```
tests/
├── agents/                          # Individual agent tests
│   ├── test_research_agent_standalone.py    # Research agent testing (START HERE)
│   ├── test_research_output_validation.py   # Research output validation
│   ├── test_content_curator.py              # Topic selection testing
│   └── test_quality_gates.py                # Quality validation testing
├── integration/                     # End-to-end pipeline tests
│   ├── test_pipeline_integration.py         # Full pipeline execution
│   └── test_database_integration.py         # Database persistence testing
└── README.md                        # This file
```

## 🚀 Quick Start

### **PRIORITY: Start with Research Agent Testing**

The pipeline issue likely stems from the research agent, so start here:

```bash
# Navigate to project root
cd /path/to/ContentRunway

# Run research agent standalone test
python tests/agents/test_research_agent_standalone.py

# Check results
cat test_results/research_agent_test_results.json
```

### **Step-by-Step Testing Sequence**

1. **Research Agent** (Most Critical)
   ```bash
   python tests/agents/test_research_agent_standalone.py
   python tests/agents/test_research_output_validation.py
   ```

2. **Content Curator**
   ```bash
   python tests/agents/test_content_curator.py
   ```

3. **Quality Gates**
   ```bash
   python tests/agents/test_quality_gates.py
   ```

4. **Database Integration**
   ```bash
   python tests/integration/test_database_integration.py
   ```

5. **Full Pipeline Integration**
   ```bash
   python tests/integration/test_pipeline_integration.py
   ```

## 📊 Expected Output Files

Each test generates detailed results:

- `test_results/research_agent_test_results.json` - Research agent performance data
- `test_results/pipeline_test_results_*.json` - Pipeline execution results
- Console logs with real-time test progress and diagnostics

## 🔍 What Each Test Does

### **Research Agent Standalone** (`test_research_agent_standalone.py`)
- ✅ Tests ResearchCoordinatorAgent with multiple queries
- ✅ Tests domain-specific agents (IT Insurance, AI, Agentic AI, etc.)
- ✅ Tests WebResearchTool functionality
- ✅ Validates output structure and content quality
- **Purpose**: Identify if research agent is working or failing

### **Research Output Validation** (`test_research_output_validation.py`)
- ✅ Validates research result structure
- ✅ Checks source and topic field requirements
- ✅ Validates score ranges and data types
- **Purpose**: Ensure research agent returns properly formatted data

### **Content Curator** (`test_content_curator.py`)
- ✅ Tests topic selection logic
- ✅ Tests scoring algorithms
- ✅ Tests edge cases (all high/low quality topics)
- **Purpose**: Verify topic curation works correctly

### **Quality Gates** (`test_quality_gates.py`)
- ✅ Tests all 4 quality validation agents individually
- ✅ Tests parallel execution performance
- ✅ Tests scoring algorithms and thresholds
- **Purpose**: Ensure quality validation is working

### **Database Integration** (`test_database_integration.py`)
- ✅ Tests topic creation and retrieval
- ✅ Tests content draft creation and retrieval
- ✅ Tests research source storage
- ✅ Tests quality assessment storage
- **Purpose**: Verify data is actually being saved to database

### **Pipeline Integration** (`test_pipeline_integration.py`)
- ✅ Tests complete end-to-end pipeline execution
- ✅ Tests step-by-step execution for debugging
- ✅ Tests error handling and recovery
- **Purpose**: Identify where the pipeline is failing

## 🐛 Troubleshooting

### **Issue: "No module named 'langgraph'"**
```bash
# Make sure you're in the right directory and Python path is set
cd /path/to/ContentRunway
export PYTHONPATH="${PYTHONPATH}:$(pwd)/langgraph"
```

### **Issue: "No module named 'backend'"**
```bash
# Make sure backend modules are accessible
export PYTHONPATH="${PYTHONPATH}:$(pwd)/backend"
```

### **Issue: Database connection errors**
```bash
# Make sure Docker services are running
docker-compose up -d
```

### **Issue: Import errors for agents**
Check if the agent files exist:
```bash
ls -la langgraph/contentrunway/agents/
```

## 📈 Success Criteria

### **Research Agent Tests Should Show:**
- ✅ 10+ sources found per domain
- ✅ 5+ topic ideas generated
- ✅ Execution time < 30 seconds per test
- ✅ No import or connection errors

### **Database Tests Should Show:**
- ✅ Topics successfully created and retrieved
- ✅ Content drafts successfully created and retrieved
- ✅ No database connection errors

### **Pipeline Tests Should Show:**
- ✅ All pipeline steps complete successfully
- ✅ Real content generated (not simulation)
- ✅ Final quality score > 0.85

## 🔧 Debugging Tips

### **If Research Agent Fails:**
1. Check if `WebResearchTool` is properly implemented
2. Check if domain agents can be imported
3. Check if API keys are set for external services
4. Look for network connectivity issues

### **If Database Tests Fail:**
1. Check database connection strings
2. Verify table schemas match model definitions
3. Check if sync database functions exist
4. Ensure Docker PostgreSQL is running

### **If Pipeline Never Executes Real Agents:**
1. Check `backend/app/tasks/pipeline_tasks.py:224` - This is where it falls back to simulation
2. Add logging to see why LangGraph execution fails
3. Check imports and dependencies for LangGraph components

## 📝 Analyzing Results

### **Research Agent Results:**
```json
{
  "test_case": 1,
  "query": "AI agents and multi-agent systems", 
  "sources_found": 15,
  "topics_generated": 8,
  "execution_time": 12.5,
  "success": true,
  "errors": []
}
```

### **Key Metrics to Watch:**
- **sources_found**: Should be > 5 per test
- **topics_generated**: Should be > 3 per test
- **execution_time**: Should be < 30s per test
- **success**: Should be `true` for most tests
- **errors**: Should be empty array

## 🎯 Next Steps After Testing

1. **Fix any failing tests** before proceeding
2. **Identify root cause** of "success but no content" issue
3. **Fix LangGraph pipeline execution** to stop falling back to simulation
4. **Re-run full pipeline** and verify content appears in database
5. **Check content tab** in frontend to confirm content is visible

## 💡 Pro Tips

- **Run tests individually** first to isolate issues
- **Check console output** for detailed error messages  
- **Save test results** for comparison after fixes
- **Use step-by-step pipeline test** for detailed debugging
- **Start with research agent** - it's the most likely failure point

---

**Need Help?** 
- Review the detailed error messages in console output
- Check the JSON result files for specific failure details
- Run tests individually to isolate specific issues
- Verify all dependencies are properly installed