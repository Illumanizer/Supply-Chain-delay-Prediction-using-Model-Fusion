"""
Enhanced Supply Chain Data Generator
====================================
Rich dataset with multi-source features for GNN + LSTM + XGBoost fusion.

Features designed so that:
- GNN benefits from network structure and node relationships
- LSTM benefits from temporal patterns
- XGBoost integrates everything

Output: 10,000 shipments with 30+ features
"""

import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

# ============== CONFIGURATION ==============
NUM_SUPPLIERS = 40
NUM_PORTS = 15
NUM_WAREHOUSES = 25
NUM_CUSTOMERS = 50
NUM_SHIPMENTS = 10000
DATE_START = datetime(2022, 1, 1)
DATE_END = datetime(2024, 6, 30)

# Geographic regions with risk profiles
REGIONS = {
    'East Asia': {'countries': ['China', 'Taiwan', 'South Korea', 'Japan'], 'risk_base': 0.3, 'infra_quality': 0.85},
    'Southeast Asia': {'countries': ['Vietnam', 'Thailand', 'Malaysia', 'Indonesia', 'Philippines'], 'risk_base': 0.45, 'infra_quality': 0.7},
    'South Asia': {'countries': ['India', 'Bangladesh'], 'risk_base': 0.5, 'infra_quality': 0.6},
    'Western Europe': {'countries': ['Germany', 'France', 'Italy', 'Spain', 'Poland', 'Czech Republic'], 'risk_base': 0.2, 'infra_quality': 0.9},
    'North America': {'countries': ['USA', 'Canada', 'Mexico'], 'risk_base': 0.25, 'infra_quality': 0.88},
    'South America': {'countries': ['Brazil'], 'risk_base': 0.45, 'infra_quality': 0.65},
    'Middle East': {'countries': ['Turkey', 'Dubai'], 'risk_base': 0.35, 'infra_quality': 0.75},
    'Oceania': {'countries': ['Australia'], 'risk_base': 0.2, 'infra_quality': 0.85},
    'UK': {'countries': ['UK'], 'risk_base': 0.22, 'infra_quality': 0.88}
}

def get_region(country):
    for region, data in REGIONS.items():
        if country in data['countries']:
            return region, data['risk_base'], data['infra_quality']
    return 'Unknown', 0.4, 0.7

# Product categories with different risk profiles
PRODUCT_CATEGORIES = {
    'Electronics': {'fragility': 0.7, 'value_density': 0.9, 'temp_sensitive': 0.3, 'customs_complexity': 0.6},
    'Machinery': {'fragility': 0.4, 'value_density': 0.7, 'temp_sensitive': 0.1, 'customs_complexity': 0.5},
    'Textiles': {'fragility': 0.2, 'value_density': 0.3, 'temp_sensitive': 0.1, 'customs_complexity': 0.3},
    'Chemicals': {'fragility': 0.5, 'value_density': 0.5, 'temp_sensitive': 0.6, 'customs_complexity': 0.8},
    'Food': {'fragility': 0.6, 'value_density': 0.4, 'temp_sensitive': 0.9, 'customs_complexity': 0.7},
    'Automotive': {'fragility': 0.3, 'value_density': 0.6, 'temp_sensitive': 0.1, 'customs_complexity': 0.5},
    'Pharmaceuticals': {'fragility': 0.8, 'value_density': 0.95, 'temp_sensitive': 0.85, 'customs_complexity': 0.9},
    'Raw Materials': {'fragility': 0.1, 'value_density': 0.2, 'temp_sensitive': 0.05, 'customs_complexity': 0.2}
}

# Coordinates
COORDS = {
    'China': (35.86, 104.19), 'Vietnam': (14.06, 108.28), 'India': (20.59, 78.96),
    'Taiwan': (23.69, 121.0), 'South Korea': (35.91, 127.77), 'Japan': (36.20, 138.25),
    'Thailand': (15.87, 100.99), 'Malaysia': (4.21, 101.98), 'Indonesia': (-0.79, 113.92),
    'Germany': (51.17, 10.45), 'Mexico': (23.63, -102.55), 'Brazil': (-14.24, -51.93),
    'Poland': (51.92, 19.15), 'Turkey': (38.96, 35.24), 'Bangladesh': (23.68, 90.36),
    'Philippines': (12.88, 121.77), 'Italy': (41.87, 12.57), 'Spain': (40.46, -3.75),
    'Czech Republic': (49.82, 15.47), 'USA': (37.09, -95.71), 'UK': (55.38, -3.44),
    'Australia': (-25.27, 133.78), 'Canada': (56.13, -106.35), 'France': (46.23, 2.21),
    'Shanghai': (31.23, 121.47), 'Shenzhen': (22.54, 114.06), 'Singapore': (1.35, 103.82),
    'Busan': (35.18, 129.08), 'Rotterdam': (51.92, 4.48), 'Los Angeles': (34.05, -118.24),
    'Long Beach': (33.77, -118.19), 'Hamburg': (53.55, 9.99), 'Antwerp': (51.22, 4.40),
    'Dubai': (25.20, 55.27), 'Chicago': (41.88, -87.63), 'Dallas': (32.78, -96.80),
    'New York': (40.71, -74.01), 'Atlanta': (33.75, -84.39), 'London': (51.51, -0.13),
    'Frankfurt': (50.11, 6.68), 'Paris': (48.86, 2.35), 'Tokyo': (35.68, 139.65),
    'Sydney': (-33.87, 151.21), 'Toronto': (43.65, -79.38), 'Miami': (25.76, -80.19),
    'Seattle': (47.61, -122.33), 'Phoenix': (33.45, -112.07), 'Denver': (39.74, -104.99),
    'Hong Kong': (22.32, 114.17), 'Kaohsiung': (22.62, 120.31), 'Yokohama': (35.44, 139.64),
    'Ningbo': (29.87, 121.54), 'Qingdao': (36.07, 120.38), 'Tianjin': (39.14, 117.18),
    'Ho Chi Minh': (10.82, 106.63), 'Bangkok': (13.76, 100.50), 'Mumbai': (19.08, 72.88),
    'Chennai': (13.08, 80.27), 'Colombo': (6.93, 79.85), 'Karachi': (24.86, 67.01)
}

SUPPLIER_COUNTRIES = ['China'] * 8 + ['Vietnam'] * 4 + ['India'] * 4 + ['Taiwan'] * 3 + \
                     ['South Korea'] * 3 + ['Japan'] * 3 + ['Thailand'] * 3 + ['Malaysia'] * 2 + \
                     ['Indonesia'] * 2 + ['Germany'] * 3 + ['Mexico'] * 2 + ['Brazil'] * 1 + \
                     ['Poland'] * 1 + ['Bangladesh'] * 1

PORT_NAMES = ['Shanghai', 'Shenzhen', 'Singapore', 'Busan', 'Rotterdam', 'Los Angeles', 
              'Long Beach', 'Hamburg', 'Antwerp', 'Dubai', 'Hong Kong', 'Ningbo', 
              'Tokyo', 'Mumbai', 'Bangkok']

WAREHOUSE_LOCATIONS = ['Los Angeles', 'Chicago', 'Dallas', 'New York', 'Atlanta',
                       'London', 'Frankfurt', 'Paris', 'Tokyo', 'Sydney',
                       'Toronto', 'Miami', 'Seattle', 'Phoenix', 'Denver',
                       'Shanghai', 'Singapore', 'Dubai', 'Mumbai', 'Mexico',
                       'Sao Paulo', 'Amsterdam', 'Madrid', 'Milan', 'Warsaw']

CUSTOMER_COUNTRIES = ['USA'] * 18 + ['Germany'] * 6 + ['UK'] * 5 + ['Japan'] * 4 + \
                     ['Australia'] * 3 + ['Canada'] * 4 + ['France'] * 4 + ['China'] * 3 + \
                     ['Brazil'] * 2 + ['Mexico'] * 1

def get_coords(location):
    if location in COORDS:
        return COORDS[location]
    return (np.random.uniform(-60, 60), np.random.uniform(-180, 180))

# ============== GENERATE NODES ==============
print("Generating enhanced nodes...")

nodes = []
node_id = 0

# Suppliers with rich features
for i in range(NUM_SUPPLIERS):
    country = SUPPLIER_COUNTRIES[i % len(SUPPLIER_COUNTRIES)]
    lat, lon = get_coords(country)
    region, risk_base, infra = get_region(country)
    
    # Supplier-specific attributes
    specialization = random.choice(list(PRODUCT_CATEGORIES.keys()))
    
    nodes.append({
        'node_id': f'SUP_{node_id:03d}',
        'node_type': 'supplier',
        'name': f'Supplier_{i+1}',
        'country': country,
        'region': region,
        'latitude': lat + np.random.uniform(-2, 2),
        'longitude': lon + np.random.uniform(-2, 2),
        'capacity': np.random.randint(2000, 15000),
        'reliability_score': np.clip(np.random.normal(0.82, 0.12), 0.5, 0.99),
        'lead_time_days': np.random.randint(3, 20),
        'quality_rating': np.clip(np.random.normal(0.8, 0.1), 0.6, 1.0),
        'years_in_operation': np.random.randint(2, 30),
        'certification_level': np.random.choice([1, 2, 3], p=[0.3, 0.5, 0.2]),  # 1=basic, 2=ISO, 3=premium
        'labor_cost_index': np.clip(np.random.normal(0.5, 0.2), 0.2, 0.9),
        'infrastructure_quality': infra + np.random.uniform(-0.1, 0.1),
        'political_stability': np.clip(1 - risk_base + np.random.uniform(-0.1, 0.1), 0.3, 1.0),
        'specialization': specialization,
        'num_product_lines': np.random.randint(1, 8),
        'avg_order_size': np.random.randint(100, 2000),
        'on_time_history': np.clip(np.random.normal(0.85, 0.1), 0.6, 0.99),
        'defect_rate': np.clip(np.random.exponential(0.02), 0.001, 0.1)
    })
    node_id += 1

# Ports with rich features
for i, port_name in enumerate(PORT_NAMES):
    lat, lon = get_coords(port_name)
    
    # Determine port region
    if port_name in ['Shanghai', 'Shenzhen', 'Hong Kong', 'Ningbo']:
        region = 'East Asia'
    elif port_name in ['Singapore', 'Bangkok']:
        region = 'Southeast Asia'
    elif port_name in ['Mumbai']:
        region = 'South Asia'
    elif port_name in ['Rotterdam', 'Hamburg', 'Antwerp']:
        region = 'Western Europe'
    elif port_name in ['Los Angeles', 'Long Beach']:
        region = 'North America'
    elif port_name in ['Dubai']:
        region = 'Middle East'
    elif port_name in ['Tokyo']:
        region = 'East Asia'
    else:
        region = 'Unknown'
    
    _, risk_base, infra = get_region(region) if region != 'Unknown' else ('Unknown', 0.3, 0.8)
    
    nodes.append({
        'node_id': f'PRT_{node_id:03d}',
        'node_type': 'port',
        'name': port_name,
        'country': port_name,
        'region': region,
        'latitude': lat,
        'longitude': lon,
        'capacity': np.random.randint(50000, 200000),
        'reliability_score': np.clip(np.random.normal(0.88, 0.08), 0.7, 0.98),
        'lead_time_days': np.random.randint(1, 5),
        'quality_rating': np.clip(np.random.normal(0.85, 0.08), 0.7, 1.0),
        'years_in_operation': np.random.randint(20, 100),
        'certification_level': np.random.choice([2, 3], p=[0.4, 0.6]),
        'labor_cost_index': np.clip(np.random.normal(0.6, 0.15), 0.3, 0.9),
        'infrastructure_quality': np.clip(infra + np.random.uniform(-0.05, 0.1), 0.7, 1.0),
        'political_stability': np.clip(1 - risk_base + np.random.uniform(-0.05, 0.1), 0.5, 1.0),
        'specialization': 'General',
        'num_product_lines': np.random.randint(5, 15),
        'avg_order_size': np.random.randint(1000, 10000),
        'on_time_history': np.clip(np.random.normal(0.88, 0.08), 0.7, 0.98),
        'defect_rate': np.clip(np.random.exponential(0.01), 0.001, 0.05),
        'berth_count': np.random.randint(10, 50),
        'crane_count': np.random.randint(20, 100),
        'annual_teu': np.random.randint(5, 50) * 1000000,  # TEU capacity
        'avg_dwell_time': np.random.uniform(2, 8),  # days
        'congestion_tendency': np.random.uniform(0.2, 0.7)
    })
    node_id += 1

# Warehouses with rich features
for i in range(NUM_WAREHOUSES):
    wh_location = WAREHOUSE_LOCATIONS[i % len(WAREHOUSE_LOCATIONS)]
    lat, lon = get_coords(wh_location)
    
    # Determine warehouse region
    region = 'Unknown'
    for r, data in REGIONS.items():
        if wh_location in data['countries'] or any(wh_location in c for c in data['countries']):
            region = r
            break
    
    _, risk_base, infra = get_region(region) if region != 'Unknown' else ('Unknown', 0.3, 0.8)
    
    nodes.append({
        'node_id': f'WH_{node_id:03d}',
        'node_type': 'warehouse',
        'name': f'{wh_location}_WH_{i+1}',
        'country': wh_location,
        'region': region,
        'latitude': lat + np.random.uniform(-1, 1),
        'longitude': lon + np.random.uniform(-1, 1),
        'capacity': np.random.randint(10000, 50000),
        'reliability_score': np.clip(np.random.normal(0.92, 0.05), 0.8, 0.99),
        'lead_time_days': np.random.randint(1, 3),
        'quality_rating': np.clip(np.random.normal(0.9, 0.05), 0.8, 1.0),
        'years_in_operation': np.random.randint(5, 40),
        'certification_level': np.random.choice([1, 2, 3], p=[0.2, 0.5, 0.3]),
        'labor_cost_index': np.clip(np.random.normal(0.6, 0.15), 0.3, 0.9),
        'infrastructure_quality': np.clip(infra + np.random.uniform(-0.05, 0.1), 0.7, 1.0),
        'political_stability': np.clip(1 - risk_base + np.random.uniform(-0.05, 0.1), 0.6, 1.0),
        'specialization': random.choice(list(PRODUCT_CATEGORIES.keys())),
        'num_product_lines': np.random.randint(3, 12),
        'avg_order_size': np.random.randint(200, 3000),
        'on_time_history': np.clip(np.random.normal(0.92, 0.05), 0.8, 0.99),
        'defect_rate': np.clip(np.random.exponential(0.005), 0.001, 0.03),
        'automation_level': np.random.uniform(0.3, 0.95),
        'cold_storage': np.random.choice([0, 1], p=[0.6, 0.4]),
        'hazmat_certified': np.random.choice([0, 1], p=[0.7, 0.3])
    })
    node_id += 1

# Customers with rich features
for i in range(NUM_CUSTOMERS):
    country = CUSTOMER_COUNTRIES[i % len(CUSTOMER_COUNTRIES)]
    lat, lon = get_coords(country)
    region, _, _ = get_region(country)
    
    nodes.append({
        'node_id': f'CUS_{node_id:03d}',
        'node_type': 'customer',
        'name': f'Customer_{i+1}',
        'country': country,
        'region': region,
        'latitude': lat + np.random.uniform(-5, 5),
        'longitude': lon + np.random.uniform(-5, 5),
        'capacity': np.random.randint(500, 8000),
        'reliability_score': 1.0,
        'lead_time_days': 0,
        'quality_rating': 1.0,
        'years_in_operation': np.random.randint(1, 50),
        'certification_level': np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2]),
        'labor_cost_index': 0,
        'infrastructure_quality': np.random.uniform(0.7, 1.0),
        'political_stability': np.random.uniform(0.7, 1.0),
        'specialization': random.choice(list(PRODUCT_CATEGORIES.keys())),
        'num_product_lines': np.random.randint(1, 10),
        'avg_order_size': np.random.randint(50, 1500),
        'on_time_history': 1.0,
        'defect_rate': 0,
        'order_frequency': np.random.choice(['daily', 'weekly', 'monthly'], p=[0.3, 0.5, 0.2]),
        'priority_level': np.random.choice([1, 2, 3], p=[0.2, 0.5, 0.3]),  # 1=low, 2=medium, 3=high
        'contract_type': np.random.choice(['spot', 'annual', 'multi-year'], p=[0.3, 0.5, 0.2])
    })
    node_id += 1

nodes_df = pd.DataFrame(nodes)

# Fill NaN for missing columns
for col in nodes_df.columns:
    if nodes_df[col].dtype == 'object':
        nodes_df[col] = nodes_df[col].fillna('Unknown')
    else:
        nodes_df[col] = nodes_df[col].fillna(0)

print(f"Created {len(nodes_df)} nodes")

# ============== GENERATE EDGES ==============
print("Generating enhanced edges...")

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lat2, lon1, lon2 = map(np.radians, [lat1, lat2, lon1, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

edges = []
edge_id = 0

suppliers = nodes_df[nodes_df['node_type'] == 'supplier']
ports = nodes_df[nodes_df['node_type'] == 'port']
warehouses = nodes_df[nodes_df['node_type'] == 'warehouse']
customers = nodes_df[nodes_df['node_type'] == 'customer']

# Supplier -> Port connections
for _, sup in suppliers.iterrows():
    port_distances = []
    for _, port in ports.iterrows():
        dist = haversine_distance(sup['latitude'], sup['longitude'], 
                                   port['latitude'], port['longitude'])
        port_distances.append((port['node_id'], port['name'], port['region'], dist))
    port_distances.sort(key=lambda x: x[3])
    
    num_connections = np.random.randint(2, 5)
    for port_id, port_name, port_region, dist in port_distances[:num_connections]:
        # Edge features
        route_type = 'road' if dist < 500 else ('rail' if dist < 2000 else 'multimodal')
        
        # Calculate border crossings based on region match
        same_region = (sup['region'] == port_region)
        border_cross = 0 if same_region else np.random.randint(1, 3)
        
        edges.append({
            'edge_id': f'E_{edge_id:04d}',
            'source': sup['node_id'],
            'target': port_id,
            'route_type': route_type,
            'distance_km': round(dist, 2),
            'base_transit_days': max(1, int(dist / 400)),
            'cost_per_unit': round(np.random.uniform(0.5, 2.5) * dist / 100, 2),
            'capacity': np.random.randint(500, 5000),
            'route_reliability': np.clip(np.random.normal(0.9, 0.08), 0.7, 0.99),
            'customs_complexity': np.random.uniform(0.1, 0.5),
            'border_crossings': border_cross,
            'alternative_routes': np.random.randint(1, 4),
            'weather_sensitivity': np.random.uniform(0.2, 0.7),
            'congestion_factor': np.random.uniform(0.1, 0.5),
            'insurance_rate': np.random.uniform(0.001, 0.01),
            'handling_complexity': np.random.uniform(0.2, 0.6)
        })
        edge_id += 1

# Port -> Port connections (sea routes)
sea_routes = [
    ('Shanghai', 'Los Angeles'), ('Shanghai', 'Long Beach'), ('Shenzhen', 'Los Angeles'),
    ('Shanghai', 'Rotterdam'), ('Shenzhen', 'Rotterdam'), ('Singapore', 'Rotterdam'),
    ('Busan', 'Los Angeles'), ('Busan', 'Long Beach'), ('Singapore', 'Dubai'),
    ('Dubai', 'Rotterdam'), ('Hamburg', 'Rotterdam'), ('Antwerp', 'Rotterdam'),
    ('Shanghai', 'Singapore'), ('Shenzhen', 'Singapore'), ('Singapore', 'Busan'),
    ('Hong Kong', 'Los Angeles'), ('Hong Kong', 'Rotterdam'), ('Ningbo', 'Rotterdam'),
    ('Tokyo', 'Los Angeles'), ('Tokyo', 'Long Beach'), ('Mumbai', 'Rotterdam'),
    ('Mumbai', 'Dubai'), ('Bangkok', 'Singapore'), ('Shanghai', 'Tokyo'),
    ('Shanghai', 'Busan'), ('Shenzhen', 'Hong Kong'), ('Rotterdam', 'Hamburg'),
    ('Los Angeles', 'Long Beach')
]

port_name_to_id = {p['name']: p['node_id'] for _, p in ports.iterrows()}

for src_name, tgt_name in sea_routes:
    if src_name in port_name_to_id and tgt_name in port_name_to_id:
        src = nodes_df[nodes_df['node_id'] == port_name_to_id[src_name]].iloc[0]
        tgt = nodes_df[nodes_df['node_id'] == port_name_to_id[tgt_name]].iloc[0]
        dist = haversine_distance(src['latitude'], src['longitude'],
                                   tgt['latitude'], tgt['longitude'])
        
        edges.append({
            'edge_id': f'E_{edge_id:04d}',
            'source': port_name_to_id[src_name],
            'target': port_name_to_id[tgt_name],
            'route_type': 'sea',
            'distance_km': round(dist, 2),
            'base_transit_days': max(5, int(dist / 600)),
            'cost_per_unit': round(np.random.uniform(1.0, 4.0) * dist / 100, 2),
            'capacity': np.random.randint(10000, 50000),
            'route_reliability': np.clip(np.random.normal(0.92, 0.05), 0.8, 0.99),
            'customs_complexity': np.random.uniform(0.3, 0.8),
            'border_crossings': 1,
            'alternative_routes': np.random.randint(1, 3),
            'weather_sensitivity': np.random.uniform(0.4, 0.9),
            'congestion_factor': np.random.uniform(0.2, 0.6),
            'insurance_rate': np.random.uniform(0.002, 0.015),
            'handling_complexity': np.random.uniform(0.3, 0.7)
        })
        edge_id += 1

# Port -> Warehouse connections
for _, wh in warehouses.iterrows():
    port_distances = []
    for _, port in ports.iterrows():
        dist = haversine_distance(wh['latitude'], wh['longitude'],
                                   port['latitude'], port['longitude'])
        if dist < 5000:  # Only connect if within reasonable distance
            port_distances.append((port['node_id'], dist))
    
    port_distances.sort(key=lambda x: x[1])
    for port_id, dist in port_distances[:3]:
        route_type = 'road' if dist < 500 else ('rail' if dist < 2000 else 'multimodal')
        
        edges.append({
            'edge_id': f'E_{edge_id:04d}',
            'source': port_id,
            'target': wh['node_id'],
            'route_type': route_type,
            'distance_km': round(dist, 2),
            'base_transit_days': max(1, int(dist / 500)),
            'cost_per_unit': round(np.random.uniform(0.3, 2.0) * dist / 100, 2),
            'capacity': np.random.randint(1000, 8000),
            'route_reliability': np.clip(np.random.normal(0.92, 0.05), 0.8, 0.99),
            'customs_complexity': np.random.uniform(0.1, 0.4),
            'border_crossings': 0,
            'alternative_routes': np.random.randint(2, 5),
            'weather_sensitivity': np.random.uniform(0.2, 0.6),
            'congestion_factor': np.random.uniform(0.1, 0.4),
            'insurance_rate': np.random.uniform(0.001, 0.008),
            'handling_complexity': np.random.uniform(0.2, 0.5)
        })
        edge_id += 1

# Warehouse -> Customer connections
for _, cus in customers.iterrows():
    wh_distances = []
    for _, wh in warehouses.iterrows():
        dist = haversine_distance(cus['latitude'], cus['longitude'],
                                   wh['latitude'], wh['longitude'])
        wh_distances.append((wh['node_id'], dist))
    
    wh_distances.sort(key=lambda x: x[1])
    num_connections = np.random.randint(1, 4)
    for wh_id, dist in wh_distances[:num_connections]:
        edges.append({
            'edge_id': f'E_{edge_id:04d}',
            'source': wh_id,
            'target': cus['node_id'],
            'route_type': 'road' if dist < 1000 else 'multimodal',
            'distance_km': round(dist, 2),
            'base_transit_days': max(1, int(dist / 600)),
            'cost_per_unit': round(np.random.uniform(0.2, 1.5) * dist / 100, 2),
            'capacity': np.random.randint(500, 3000),
            'route_reliability': np.clip(np.random.normal(0.94, 0.04), 0.85, 0.99),
            'customs_complexity': np.random.uniform(0.05, 0.3),
            'border_crossings': 0,
            'alternative_routes': np.random.randint(2, 6),
            'weather_sensitivity': np.random.uniform(0.1, 0.5),
            'congestion_factor': np.random.uniform(0.1, 0.4),
            'insurance_rate': np.random.uniform(0.0005, 0.005),
            'handling_complexity': np.random.uniform(0.1, 0.4)
        })
        edge_id += 1

edges_df = pd.DataFrame(edges)
print(f"Created {len(edges_df)} edges")

# ============== BUILD GRAPH AND CALCULATE METRICS ==============
print("Building graph and calculating centrality metrics...")

G = nx.DiGraph()
for _, node in nodes_df.iterrows():
    G.add_node(node['node_id'], **node.to_dict())
for _, edge in edges_df.iterrows():
    G.add_edge(edge['source'], edge['target'], **edge.to_dict())

# Calculate various centrality metrics
degree_centrality = nx.degree_centrality(G)
in_degree_centrality = nx.in_degree_centrality(G)
out_degree_centrality = nx.out_degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)
closeness_centrality = nx.closeness_centrality(G)
pagerank = nx.pagerank(G)

try:
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
except:
    eigenvector_centrality = {n: 0 for n in G.nodes()}

# Add to nodes_df
nodes_df['degree_centrality'] = nodes_df['node_id'].map(degree_centrality)
nodes_df['in_degree_centrality'] = nodes_df['node_id'].map(in_degree_centrality)
nodes_df['out_degree_centrality'] = nodes_df['node_id'].map(out_degree_centrality)
nodes_df['betweenness_centrality'] = nodes_df['node_id'].map(betweenness_centrality)
nodes_df['closeness_centrality'] = nodes_df['node_id'].map(closeness_centrality)
nodes_df['pagerank'] = nodes_df['node_id'].map(pagerank)
nodes_df['eigenvector_centrality'] = nodes_df['node_id'].map(eigenvector_centrality)
nodes_df['in_degree'] = nodes_df['node_id'].map(dict(G.in_degree()))
nodes_df['out_degree'] = nodes_df['node_id'].map(dict(G.out_degree()))

# Find valid paths
print("Finding valid supply chain paths...")
valid_paths = []
for _, sup in suppliers.iterrows():
    for _, cus in customers.iterrows():
        try:
            paths = list(nx.all_simple_paths(G, sup['node_id'], cus['node_id'], cutoff=6))
            for path in paths[:3]:
                if len(path) >= 3:
                    valid_paths.append(path)
        except nx.NetworkXNoPath:
            continue

print(f"Found {len(valid_paths)} valid supply chain paths")

# ============== GENERATE SHIPMENTS ==============
print("Generating enhanced shipments...")

date_range = pd.date_range(DATE_START, DATE_END, freq='D')

# Pre-generate hidden factors
HIDDEN_WEATHER = np.random.normal(0, 0.08, NUM_SHIPMENTS)
HIDDEN_CONGESTION = np.random.normal(0, 0.08, NUM_SHIPMENTS)
HIDDEN_DEMAND = np.random.normal(0, 0.1, NUM_SHIPMENTS)

# Weather function
def get_weather_severity(date, location_lat, region):
    month = date.month
    
    # Seasonal pattern
    if location_lat > 20:
        seasonal = 0.3 + 0.3 * np.cos((month - 1) * np.pi / 6)
    else:
        seasonal = 0.3 + 0.2 * np.cos((month - 7) * np.pi / 6)
    
    # Monsoon effect for Asia
    if region in ['Southeast Asia', 'South Asia'] and month in [6, 7, 8, 9]:
        seasonal += 0.3
    
    # Hurricane season for Americas
    if region == 'North America' and month in [8, 9, 10]:
        seasonal += 0.2
    
    # Random severe events
    base = np.random.beta(2, 5)
    if np.random.random() < 0.03:
        base = np.random.uniform(0.7, 1.0)
    
    return np.clip(base + seasonal * 0.5, 0, 1)

# Port congestion function
port_base_congestion = {port_id: np.random.uniform(0.2, 0.5) for port_id in ports['node_id']}

def get_port_congestion(port_id, date):
    base = port_base_congestion.get(port_id, 0.3)
    
    # Q4 surge
    if date.month in [10, 11, 12]:
        base *= 1.4
    # Chinese New Year effect
    if date.month == 2:
        base *= 1.3
    # Random spikes
    if np.random.random() < 0.08:
        base = min(1.0, base + np.random.uniform(0.2, 0.5))
    
    return np.clip(base, 0, 1)

# Generate shipments
shipments = []
shipment_id = 0

for idx in range(NUM_SHIPMENTS):
    path = random.choice(valid_paths)
    date = random.choice(date_range)
    
    # Path analysis
    total_distance = 0
    base_transit_days = 0
    path_reliability = 1.0
    total_customs_complexity = 0
    total_handling_complexity = 0
    num_sea_legs = 0
    num_border_crossings = 0
    
    for i in range(len(path) - 1):
        edge_data = G.get_edge_data(path[i], path[i+1])
        if edge_data:
            total_distance += edge_data['distance_km']
            base_transit_days += edge_data['base_transit_days']
            path_reliability *= edge_data['route_reliability']
            total_customs_complexity += edge_data['customs_complexity']
            total_handling_complexity += edge_data['handling_complexity']
            num_border_crossings += edge_data['border_crossings']
            if edge_data['route_type'] == 'sea':
                num_sea_legs += 1
    
    # Source and destination info
    source_node = nodes_df[nodes_df['node_id'] == path[0]].iloc[0]
    dest_node = nodes_df[nodes_df['node_id'] == path[-1]].iloc[0]
    
    # Get intermediate nodes info
    ports_in_path = [n for n in path if n.startswith('PRT_')]
    warehouses_in_path = [n for n in path if n.startswith('WH_')]
    
    # Calculate path-based metrics
    path_betweenness = np.mean([betweenness_centrality.get(n, 0) for n in path])
    path_pagerank = np.mean([pagerank.get(n, 0) for n in path])
    max_node_risk = max([1 - nodes_df[nodes_df['node_id'] == n].iloc[0]['reliability_score'] for n in path])
    
    # Port congestion
    max_congestion = 0
    for port_id in ports_in_path:
        max_congestion = max(max_congestion, get_port_congestion(port_id, date))
    
    # Weather
    weather_severity = get_weather_severity(date, source_node['latitude'], source_node['region'])
    
    # Product and shipment characteristics
    product_category = random.choice(list(PRODUCT_CATEGORIES.keys()))
    product_info = PRODUCT_CATEGORIES[product_category]
    
    volume = np.random.randint(50, 3000)
    value_usd = volume * np.random.uniform(10, 500) * product_info['value_density']
    priority_level = np.random.choice([1, 2, 3], p=[0.3, 0.5, 0.2])
    
    # Time-based features
    demand_surge = 1.0 if date.month in [11, 12] else (0.7 if date.month in [9, 10] else 0.4)
    demand_surge += np.random.uniform(-0.15, 0.15)
    
    is_peak_season = 1 if date.month in [10, 11, 12] else 0
    is_holiday = 1 if (date.month == 12 and date.day > 15) or (date.month == 2 and date.day < 20) else 0
    
    # Upstream risk
    upstream_risk = np.random.uniform(0, 0.35)
    
    # ============== LABEL COMPUTATION ==============
    # Now depends on MULTIPLE factors including graph structure
    
    # True values with hidden noise
    weather_true = np.clip(weather_severity + HIDDEN_WEATHER[idx], 0, 1)
    congestion_true = np.clip(max_congestion + HIDDEN_CONGESTION[idx], 0, 1)
    demand_true = np.clip(demand_surge + HIDDEN_DEMAND[idx], 0, 1.5)
    
    # Compute risk score
    score = 0
    
    # Weather impact (weight: 2)
    if weather_true > 0.5:
        score += 2
    
    # Congestion impact (weight: 2)
    if congestion_true > 0.5:
        score += 2
    
    # Supplier reliability (weight: 2)
    if source_node['reliability_score'] < 0.75:
        score += 2
    
    # Path complexity - GNN can learn this! (weight: 2)
    if path_betweenness > 0.05:  # High betweenness = critical path
        score += 1
    if len(path) > 5:  # Long paths more risky
        score += 1
    
    # Sea transport risk (weight: 1)
    if num_sea_legs >= 2:
        score += 1
    
    # Border crossing complexity (weight: 1)
    if num_border_crossings >= 2:
        score += 1
    
    # Product sensitivity (weight: 1)
    if product_info['fragility'] > 0.6 or product_info['temp_sensitive'] > 0.6:
        score += 1
    
    # Customs complexity (weight: 1)
    if total_customs_complexity > 1.5:
        score += 1
    
    # Upstream cascade risk (weight: 1.5)
    if upstream_risk > 0.2:
        score += 1.5
    
    # Demand surge + season compound (weight: 1)
    if demand_true > 0.85 and is_peak_season:
        score += 1
    
    # Distance impact (weight: 0.5)
    if total_distance > 15000:
        score += 0.5
    
    # Path reliability (GNN relevant)
    if path_reliability < 0.7:
        score += 1
    
    # Determine delay
    is_delayed = 1 if score >= 6 else 0
    delay_days = max(1, int(score * 0.6) + np.random.randint(-1, 3)) if is_delayed else 0
    
    shipments.append({
        'shipment_id': f'SHP_{shipment_id:05d}',
        'date': date.strftime('%Y-%m-%d'),
        'source_node': path[0],
        'destination_node': path[-1],
        'path': '->'.join(path),
        'path_length': len(path),
        
        # Distance and time
        'total_distance_km': round(total_distance, 2),
        'planned_transit_days': base_transit_days,
        'actual_transit_days': base_transit_days + delay_days,
        'delay_days': delay_days,
        'delayed': is_delayed,
        
        # Volume and value
        'volume': volume,
        'value_usd': round(value_usd, 2),
        'priority_level': priority_level,
        
        # Product characteristics
        'product_category': product_category,
        'fragility': product_info['fragility'],
        'temp_sensitive': product_info['temp_sensitive'],
        'customs_complexity_product': product_info['customs_complexity'],
        
        # Route characteristics
        'num_sea_legs': num_sea_legs,
        'num_border_crossings': num_border_crossings,
        'path_reliability': round(path_reliability, 4),
        'total_customs_complexity': round(total_customs_complexity, 3),
        'total_handling_complexity': round(total_handling_complexity, 3),
        
        # Graph-derived features (GNN can enhance these)
        'path_betweenness': round(path_betweenness, 6),
        'path_pagerank': round(path_pagerank, 6),
        'max_node_risk': round(max_node_risk, 4),
        
        # Environmental factors
        'weather_severity': round(weather_severity, 3),
        'port_congestion': round(max_congestion, 3),
        
        # Supplier characteristics
        'supplier_reliability': round(source_node['reliability_score'], 3),
        'supplier_quality': round(source_node['quality_rating'], 3),
        'supplier_infra': round(source_node['infrastructure_quality'], 3),
        
        # Demand factors
        'demand_surge': round(demand_surge, 3),
        'is_peak_season': is_peak_season,
        'is_holiday': is_holiday,
        
        # Risk factors
        'upstream_risk': round(upstream_risk, 3),
        
        # Time features
        'month': date.month,
        'day_of_week': date.weekday(),
        'quarter': (date.month - 1) // 3 + 1,
        'week_of_year': date.isocalendar()[1]
    })
    shipment_id += 1

shipments_df = pd.DataFrame(shipments)
print(f"Created {len(shipments_df)} shipments")

# ============== GENERATE ENHANCED TIME SERIES ==============
print("Generating enhanced time series...")

time_series = []
for node_id in nodes_df['node_id']:
    node_shipments = shipments_df[
        (shipments_df['source_node'] == node_id) | 
        (shipments_df['destination_node'] == node_id)
    ]
    
    for date in pd.date_range(DATE_START, DATE_END, freq='W'):
        week_shipments = node_shipments[
            (pd.to_datetime(node_shipments['date']) >= date) &
            (pd.to_datetime(node_shipments['date']) < date + timedelta(days=7))
        ]
        
        if len(week_shipments) > 0:
            time_series.append({
                'node_id': node_id,
                'week_start': date.strftime('%Y-%m-%d'),
                'shipment_count': len(week_shipments),
                'total_volume': week_shipments['volume'].sum(),
                'total_value': week_shipments['value_usd'].sum(),
                'avg_delay': week_shipments['delay_days'].mean(),
                'max_delay': week_shipments['delay_days'].max(),
                'delay_rate': week_shipments['delayed'].mean(),
                'delay_variance': week_shipments['delay_days'].var() if len(week_shipments) > 1 else 0,
                'avg_weather': week_shipments['weather_severity'].mean(),
                'max_weather': week_shipments['weather_severity'].max(),
                'avg_congestion': week_shipments['port_congestion'].mean(),
                'max_congestion': week_shipments['port_congestion'].max(),
                'avg_path_length': week_shipments['path_length'].mean(),
                'avg_distance': week_shipments['total_distance_km'].mean(),
                'high_priority_ratio': (week_shipments['priority_level'] == 3).mean(),
                'sea_leg_ratio': (week_shipments['num_sea_legs'] > 0).mean()
            })
        else:
            time_series.append({
                'node_id': node_id,
                'week_start': date.strftime('%Y-%m-%d'),
                'shipment_count': 0,
                'total_volume': 0,
                'total_value': 0,
                'avg_delay': 0,
                'max_delay': 0,
                'delay_rate': 0,
                'delay_variance': 0,
                'avg_weather': 0,
                'max_weather': 0,
                'avg_congestion': 0,
                'max_congestion': 0,
                'avg_path_length': 0,
                'avg_distance': 0,
                'high_priority_ratio': 0,
                'sea_leg_ratio': 0
            })

time_series_df = pd.DataFrame(time_series)
print(f"Created {len(time_series_df)} time series records")

# ============== SAVE DATA ==============
print("\nSaving datasets...")

nodes_df.to_csv('supply_chain_nodes.csv', index=False)
edges_df.to_csv('supply_chain_edges.csv', index=False)
shipments_df.to_csv('supply_chain_shipments.csv', index=False)
time_series_df.to_csv('supply_chain_timeseries.csv', index=False)

# ============== SUMMARY ==============
print("\n" + "="*60)
print("ENHANCED DATASET GENERATION COMPLETE")
print("="*60)

print(f"\n📊 NODES: {len(nodes_df)}")
print(f"   - Suppliers: {len(nodes_df[nodes_df['node_type'] == 'supplier'])}")
print(f"   - Ports: {len(nodes_df[nodes_df['node_type'] == 'port'])}")
print(f"   - Warehouses: {len(nodes_df[nodes_df['node_type'] == 'warehouse'])}")
print(f"   - Customers: {len(nodes_df[nodes_df['node_type'] == 'customer'])}")
print(f"   - Features per node: {len(nodes_df.columns)}")

print(f"\n🔗 EDGES: {len(edges_df)}")
print(f"   - Road: {len(edges_df[edges_df['route_type'] == 'road'])}")
print(f"   - Rail: {len(edges_df[edges_df['route_type'] == 'rail'])}")
print(f"   - Sea: {len(edges_df[edges_df['route_type'] == 'sea'])}")
print(f"   - Multimodal: {len(edges_df[edges_df['route_type'] == 'multimodal'])}")
print(f"   - Features per edge: {len(edges_df.columns)}")

print(f"\n📦 SHIPMENTS: {len(shipments_df)}")
delayed = shipments_df['delayed'].sum()
print(f"   - Delayed: {delayed} ({100*delayed/len(shipments_df):.1f}%)")
print(f"   - On-time: {len(shipments_df) - delayed} ({100*(len(shipments_df)-delayed)/len(shipments_df):.1f}%)")
print(f"   - Features per shipment: {len(shipments_df.columns)}")

print(f"\n📈 TIME SERIES: {len(time_series_df)} weekly records")
print(f"   - Features per record: {len(time_series_df.columns)}")

print("\n✅ Files saved:")
print("   - supply_chain_nodes.csv")
print("   - supply_chain_edges.csv")
print("   - supply_chain_shipments.csv")
print("   - supply_chain_timeseries.csv")

print("\n🎯 KEY FEATURES FOR EACH MODEL:")
print("   GNN: Node centrality, path reliability, network structure")
print("   LSTM: Weekly patterns, delay trends, congestion history")
print("   XGBoost: Product type, weather, congestion, customs complexity")