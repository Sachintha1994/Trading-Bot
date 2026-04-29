"""
Binance Trading Bot — AI Analyst
================================
Uses Google Gemini 1.5 Flash to provide natural language reasoning
and confidence scores for trade signals.
"""

import json
import sys
import os
import google.generativeai as genai
import pandas as pd

# Add parent directory to path to allow direct execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import config
    from core.utils import get_logger
except ImportError:
    import config
    from utils import get_logger

logger = get_logger("AIAnalyst")

class AIAnalyst:
    """
    Interfaces with Gemini API to analyze technical data.
    """

    def __init__(self):
        self.enabled = config.LLM_ANALYST_ENABLED and bool(config.GEMINI_API_KEY)
        if self.enabled:
            try:
                genai.configure(api_key=config.GEMINI_API_KEY)
                
                # Auto-detect best available model
                available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                
                # Try to find a match or fallback
                target = f"models/{config.LLM_MODEL_NAME}"
                if target in available_models:
                    self.model_name = target
                elif "models/gemini-1.5-flash" in available_models:
                    self.model_name = "models/gemini-1.5-flash"
                elif "models/gemini-pro" in available_models:
                    self.model_name = "models/gemini-pro"
                elif available_models:
                    self.model_name = available_models[0]
                else:
                    raise Exception("No generative models found in your account.")

                self.model = genai.GenerativeModel(self.model_name)
                logger.info("🧠 AI Analyst initialized using %s", self.model_name)
            except Exception as e:
                logger.error("❌ AI Analyst initialization failed: %s", e)
                self.enabled = False
        else:
            logger.warning("🧠 AI Analyst is DISABLED (Check config or API key)")

    def analyze_trade(
        self,
        symbol: str,
        side: str,
        current_price: float,
        df: pd.DataFrame,
        technical_reasons: list[str],
        confluence_score: float
    ) -> dict:
        """
        Send technical data to Gemini for a detailed AI assessment.
        """
        if not self.enabled:
            return {"confidence": 1.0, "reasoning": "AI Analyst disabled.", "action": side}

        latest = df.iloc[-1]
        
        # Prepare data for the prompt
        tech_data = {
            "Symbol": symbol,
            "Proposed Action": side,
            "Current Price": f"${current_price:,.2f}",
            "Confluence Score": f"{confluence_score:.4f}",
            "RSI": f"{latest.get('rsi', 'N/A'):.2f}",
            "MACD Histogram": f"{latest.get('MACD_histogram', 'N/A'):.4f}",
            "Bollinger Band %": f"{latest.get('BB_pct', 'N/A'):.2f}",
            "Technical Signals": technical_reasons
        }

        prompt = f"""
        You are a professional crypto trading analyst. 
        Review the following technical analysis data for {symbol} and provide your final verdict on a {side} trade.

        TECHNICAL DATA:
        {json.dumps(tech_data, indent=2)}

        TASKS:
        1. Evaluate the strength of this {side} signal.
        2. Provide a brief 2-3 sentence reasoning.
        3. Assign a confidence score between 0.0 and 1.0.
        4. Suggest a final 'Action': BUY, SELL, or HOLD.

        Return ONLY a JSON object in this format:
        {{
            "action": "BUY/SELL/HOLD",
            "confidence": 0.85,
            "reasoning": "Your explanation here"
        }}
        """

        try:
            response = self.model.generate_content(prompt)
            # Handle possible markdown formatting in response
            text = response.text.strip().replace("```json", "").replace("```", "")
            result = json.loads(text)
            
            logger.info("🧠 AI Verdict: %s (Confidence: %.2f)", result.get("action"), result.get("confidence", 0))
            logger.info("   Reasoning: %s", result.get("reasoning"))
            
            return result
        except Exception as e:
            logger.error("❌ AI Analyst failed: %s", e)
            return {"confidence": 0.5, "reasoning": f"AI Analysis failed: {str(e)}", "action": side}

def get_analyst():
    """Singleton pattern for the analyst."""
    if not hasattr(get_analyst, "_instance"):
        get_analyst._instance = AIAnalyst()
    return get_analyst._instance

if __name__ == "__main__":
    # Quick connectivity test
    from dotenv import load_dotenv
    load_dotenv()
    
    print("Testing AI Analyst Connectivity...")
    analyst = AIAnalyst()
    if analyst.enabled:
        # Mock data for testing
        mock_df = pd.DataFrame({"rsi": [35.0], "MACD_histogram": [0.001], "BB_pct": [0.1], "close": [65000.0]})
        mock_df.index = [pd.Timestamp.now()]
        
        result = analyst.analyze_trade(
            symbol=config.TRADING_PAIR,
            side="BUY",
            current_price=65000.0,
            df=mock_df,
            technical_reasons=["RSI is near oversold", "MACD crossing up"],
            confluence_score=0.68
        )
        print("\n--- AI RESPONSE ---")
        print(json.dumps(result, indent=2))
        print("-------------------\n")
        if "action" in result:
            print("SUCCESS: Gemini API is working correctly!")
    else:
        print("AI Analyst is disabled. Check your GEMINI_API_KEY in .env")
