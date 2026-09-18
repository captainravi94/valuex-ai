from typing import Dict, Any

class FootnoteDisclosuresEngine:
    @staticmethod
    def drilldown_note(note_category: str, company_name: str, context_data: Dict[str, Any]) -> Dict[str, Any]:
        if "other income" in note_category.lower():
            total = context_data.get("other_income", 0.0)
            return {
                "note_title": "Note 24: Other Non-Operating Income Breakdown",
                "reported_total": total,
                "accounting_treatment": "Ind AS 1 / US GAAP Presentation of Income Schedules",
                "components": [
                    {"Component": "Interest Income on Bank Deposits & Treasury", "Amount": total * 0.45},
                    {"Component": "Net Gain on Derecognition of Financial Liabilities", "Amount": total * 0.35},
                    {"Component": "Foreign Exchange Fluctuations & Miscellaneous Credits", "Amount": total * 0.20}
                ],
                "auditor_observation": "Confirm recurrence of liability derecognitions before modeling this income forward."
            }
        else:
            debt = context_data.get("total_borrowings", 0.0)
            return {
                "note_title": "Note 18: Borrowings & Capital Structure Maturity",
                "reported_total": debt,
                "accounting_treatment": "Carried at Amortized Cost (Ind AS 109)",
                "components": [
                    {"Component": "Long-Term Term Loans & Notes", "Amount": debt * 0.70},
                    {"Component": "Short-Term Working Capital Facilities", "Amount": debt * 0.20},
                    {"Component": "Finance Lease Obligations", "Amount": debt * 0.10}
                ],
                "auditor_observation": "Assess liquidity reserves against near-term maturing principal obligations."
            }