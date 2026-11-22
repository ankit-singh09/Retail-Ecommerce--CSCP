
# Data Dictionary

| Column | Description |
|---|---|
| Invoice | Invoice number. Starts with "C" if it is a cancellation |
| StockCode | Product code |
| Description | Product name |
| Quantity | Units per line (negative for returns) |
| InvoiceDate | Date and time of the invoice |
| Price | Unit price in GBP |
| Customer ID | Customer identifier (missing for guest checkouts) |
| Country | Customer country |
| TotalAmount | Quantity x Price (engineered) |
| Is_Cancelled | True if the invoice is a cancellation (engineered) |
| Is_Registered | True if the row has a real Customer ID (engineered) |
