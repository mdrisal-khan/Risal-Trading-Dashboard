# Helper function to auto-retry across endpoints + Public Proxy Fallback
def binance_request(endpoint_path, params=None, timeout=15):
    # Standard official endpoints
    endpoints = [
        f"https://api1.binance.com{endpoint_path}",
        f"https://api2.binance.com{endpoint_path}",
        f"https://api3.binance.com{endpoint_path}",
        f"https://data.binance.com{endpoint_path}",
        # Public Cors Proxy Fallback for Streamlit Cloud
        f"https://corsproxy.io/?{quote(f'https://api.binance.com{endpoint_path}')}"
    ]

    for url in endpoints:
        try:
            r = SESSION.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
        except Exception:
            continue
            
    raise requests.RequestException("All Binance API endpoints failed or are blocked by host IP.")
