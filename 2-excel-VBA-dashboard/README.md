# Dynamic Sales Analytics Dashboard (Excel + VBA)

## Dashboard Overview

![Dashboard view](screenshots/dynamic-dashboard.png)

An interactive Excel dashboard for competitor and segment-level sales analysis, built on transactional simulation data.  

The workbook transforms raw exports into a navigable analytics interface with dynamic filtering, KPI/competitor comparison, and automated visual updates.

---

## What it does

- Filters sales data by product and market via dropdown selectors
- Calculates segment-level KPIs:
  - Total revenue
  - Units sold  
  - Average price  
  - Number of orders  
  - On-time delivery rate  
  - Average delay  
- Enables side-by-side comparison against selected competitors
- Updates PivotCharts and time-series visuals dynamically
- Synchronizes KPI tile colors with chart series automatically

---

## How it works

- Structured transaction data imported to Excel (uses a subset of data; approx. 3000 rows)
- PivotTables & functions aggregate revenue, volume, pricing, and delivery metrics
- VBA event listener triggers automation when selectors change
- Pivot filtering implemented programmatically (Data Model–compatible)
- Font contrast adjusted dynamically for readability
- Programmatic resizing and rerendering of a normal Excel-chart

---

## Tools Used

Excel, VBA (event-driven automation), Pivot Tables & Pivot Charts, Data Model, Functions
