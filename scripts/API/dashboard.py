import dash
import dash_core_components as dcc
import dash_html_components as html
import requests
import pandas as pd
import plotly.express as px

class FraudDashboard:
    def __init__(self, api_url):
        """Initialize Dash app & define layout."""
        self.api_url = api_url
        self.app = dash.Dash(__name__)
        self.app.layout = self.create_layout()
        self.data = self.fetch_data()

    def fetch_data(self):
        """Fetch fraud data from Flask API."""
        summary = requests.get(f"{self.api_url}/summary").json()
        fraud_trends = requests.get(f"{self.api_url}/fraud-trends").json()

        return {
            "summary": summary,
            "fraud_trends": pd.DataFrame(fraud_trends.items(), columns=["Date", "Fraud Cases"])
        }

    def create_layout(self):
        """Create dashboard layout."""
        return html.Div(children=[
            html.H1("Fraud Detection Dashboard", style={"textAlign": "center"}),

            html.Div([
                html.P(f"Total Transactions: {self.data['summary']['total_transactions']}"),
                html.P(f"Fraud Cases: {self.data['summary']['fraud_cases']}"),
                html.P(f"Fraud Percentage: {self.data['summary']['fraud_percentage']}%")
            ], style={"textAlign": "center", "fontSize": "20px"}),

            dcc.Graph(
                id="fraud_trends_chart",
                figure=px.line(
                    self.data["fraud_trends"],
                    x="Date",
                    y="Fraud Cases",
                    title="Fraud Trends Over Time"
                )
            )
        ])

    def run(self):
        """Run the Dash app."""
        self.app.run_server(debug=True, host="0.0.0.0", port=8050)

if __name__ == "__main__":
    dashboard = FraudDashboard(api_url="http://127.0.0.1:5000")
    dashboard.run()
