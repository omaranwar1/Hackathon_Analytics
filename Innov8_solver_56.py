#!/usr/bin/env python3
"""
🐛 DEBUG VERSION - Innov8 Solver v43 with Detailed Logging

Comprehensive debugging output shows:
- Inventory reservation status per warehouse
- Order assignment details
- Vehicle selection rationale
- Fulfillment progress after each phase
- Iteration-by-iteration improvements
"""

from robin_logistics import LogisticsEnvironment
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import heapq
import math

# ═══════════════════════════════════════════════════════════════════
# DEBUGGING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
DEBUG = True  # Master debug switch

# ═══════════════════════════════════════════════════════════════════
# MERGED PARAMETERS - Best from Both Solvers
# ═══════════════════════════════════════════════════════════════════

# Safety factor for capacity to avoid floating point precision issues (0-1)
CAPACITY_SAFETY = 0.999

# Target fulfillment rate before stopping vehicle additions (0-1)
# From copy: 0.92 for better reliability (vs 0.90 from main)
FULFILLMENT_TARGET = 0.92

# Maximum vehicles to use per warehouse in initial packing
MAX_VEHICLES_PER_WAREHOUSE = 4

# Maximum vehicles to add per warehouse in fallback rounds
# From copy: 2 for more conservative growth (vs 3 from main)
MAX_FALLBACK_VEHICLES_PER_WH = 2

# Maximum iterations for fallback vehicle additions
# From copy: 6 for better coverage (vs 4 from main)
MAX_FALLBACK_ITERATIONS = 6

# COST OPTIMIZATION: Prefer smaller vehicles (LightVan £300 vs Medium £625 vs Heavy £1200)
# NEW from copy
PREFER_SMALL_VEHICLES = True


def my_solver(env: LogisticsEnvironment) -> Dict:
    """
    Generate optimized solution minimizing vehicles and cost.
    DEBUG VERSION - prints detailed progress information.

    Args:
        env: LogisticsEnvironment instance

    Returns:
        Solution dict: {'routes': [{'vehicle_id': str, 'steps': [...]}]}
    """
    order_ids = env.get_all_order_ids()
    if not order_ids:
        return {"routes": []}

    warehouses = env.warehouses
    all_vehicles = env.get_all_vehicles()
    road_network = env.get_road_network_data()

    if DEBUG:
        print("\n" + "="*80)
        print("🐛 DEBUG: SOLVER STARTED")
        print("="*80)
        print(f"📦 Total Orders: {len(order_ids)}")
        print(f"🏭 Warehouses: {len(warehouses)}")
        print(f"🚛 Available Vehicles: {len(all_vehicles)}")

        # Show initial inventory
        print(f"\n📊 Initial Inventory:")
        for wh_id, wh in warehouses.items():
            print(f"  {wh_id}: {dict(wh.inventory)}")

    # Build adjacency list with distances
    adjacency_list = build_adjacency_list(road_network)

    # Calculate order requirements and locations
    if DEBUG:
        print(f"\n🔍 PHASE 1: Analyzing Orders and Assigning Warehouses...")

    order_data, reserved_inventory = analyze_orders(env, order_ids, warehouses, adjacency_list)

    if DEBUG:
        # Count orders per warehouse
        warehouse_assignments = defaultdict(int)
        split_count = 0
        unfulfillable_count = 0

        for oid, data in order_data.items():
            if data.get('requires_split'):
                split_count += 1
            elif data.get('unfulfillable'):
                unfulfillable_count += 1
            elif data.get('warehouse_id'):
                warehouse_assignments[data['warehouse_id']] += 1

        print(f"\n✅ Order Assignment Complete:")
        for wh_id, count in warehouse_assignments.items():
            print(f"  {wh_id}: {count} orders")
        if split_count > 0:
            print(f"  Split Delivery: {split_count} orders")
        if unfulfillable_count > 0:
            print(f"  ⚠️  Unfulfillable: {unfulfillable_count} orders")

    # MULTI-TIER STRATEGY: Ultra-aggressive packing into 3-5 vehicles + robust fallbacks
    # GOAL: 1-5 vehicles with >80% fulfillment

    # CRITICAL FIX: Use reserved_inventory from analyze_orders instead of fresh inventory
    # This ensures inventory tracking stays consistent between assignment and packing phases
    inventory_remaining = reserved_inventory

    # TIER 1: SMART PACKING - Choose vehicle size based on cargo volume
    # Calculate total cargo per warehouse to inform vehicle selection
    cargo_by_warehouse = calculate_cargo_by_warehouse(order_ids, order_data, warehouses)

    if DEBUG:
        print(f"\n🚛 PHASE 2: Smart Vehicle Packing (Max {MAX_VEHICLES_PER_WAREHOUSE} vehicles/warehouse)...")
        print(f"\n📊 Cargo Analysis:")
        for wh_id, cargo in cargo_by_warehouse.items():
            print(f"  {wh_id}: {cargo['order_count']} orders, {cargo['volume']:.2f}m³ total volume")

    solution, used_vehicles, assigned_orders = pack_into_limited_vehicles(
        env, order_ids, order_data, warehouses, all_vehicles,
        inventory_remaining, adjacency_list, cargo_by_warehouse,
        max_vehicles_per_warehouse=MAX_VEHICLES_PER_WAREHOUSE
    )

    # Calculate fulfillment rate
    fulfillment_rate = len(assigned_orders) / len(order_ids) if order_ids else 1.0

    if DEBUG:
        print(f"\n✅ Initial Packing Complete:")
        print(f"  Routes Created: {len(solution['routes'])}")
        print(f"  Orders Fulfilled: {len(assigned_orders)}/{len(order_ids)} ({fulfillment_rate*100:.1f}%)")
        
        # Highlight if we've reached target
        if fulfillment_rate >= FULFILLMENT_TARGET:
            print(f"  🎯 TARGET REACHED! ({fulfillment_rate*100:.1f}% >= {FULFILLMENT_TARGET*100:.0f}%)")
        else:
            gap = FULFILLMENT_TARGET - fulfillment_rate
            print(f"  📊 Gap to target: {gap*100:.1f}% ({int(gap * len(order_ids))} orders)")
        
        print(f"  Vehicles Used: {len(used_vehicles)}")

        # Show which vehicles were used
        vehicle_types = defaultdict(int)
        for route in solution['routes']:
            vehicle = env.get_vehicle_by_id(route['vehicle_id'])
            vehicle_types[vehicle.type] += 1

        print(f"\n  Vehicle Mix:")
        for vtype, count in sorted(vehicle_types.items()):
            print(f"    {vtype}: {count}")

        # Show remaining inventory
        print(f"\n📦 Inventory After Initial Packing:")
        for wh_id in warehouses.keys():
            print(f"  {wh_id}: {dict(inventory_remaining[wh_id])}")

    # TIER 2: ROBUST FALLBACK - Add vehicles until target fulfillment reached
    iteration = 0

    if DEBUG and fulfillment_rate < FULFILLMENT_TARGET:
        print(f"\n🔄 PHASE 3: Fallback Iterations (Target: {FULFILLMENT_TARGET*100:.0f}% fulfillment)...")
        print(f"  Currently at {fulfillment_rate*100:.1f}%, need {len(order_ids) - len(assigned_orders)} more orders")

    while fulfillment_rate < FULFILLMENT_TARGET and iteration < MAX_FALLBACK_ITERATIONS:
        unfulfilled = set(order_ids) - assigned_orders

        if not unfulfilled:
            break

        if DEBUG:
            print(f"\n  🔄 Iteration {iteration + 1}/{MAX_FALLBACK_ITERATIONS}:")
            print(f"    Unfulfilled orders: {len(unfulfilled)}")

        # Pack unfulfilled orders into additional vehicles (more aggressive)
        fallback_routes, fallback_used, newly_assigned = pack_fallback_orders(
            env, list(unfulfilled), order_data, warehouses, all_vehicles,
            used_vehicles, inventory_remaining, adjacency_list
        )

        if fallback_routes:
            solution["routes"].extend(fallback_routes)
            used_vehicles.update(fallback_used)
            assigned_orders.update(newly_assigned)
            old_fulfillment = fulfillment_rate
            fulfillment_rate = len(assigned_orders) / len(order_ids)

            if DEBUG:
                improvement = fulfillment_rate - old_fulfillment
                print(f"    ✅ Added {len(fallback_routes)} route(s), {len(newly_assigned)} orders fulfilled")
                print(f"    Fulfillment: {old_fulfillment*100:.1f}% → {fulfillment_rate*100:.1f}% (+{improvement*100:.1f}%)")

                # Show which vehicles were added
                for route_id in fallback_routes:
                    vehicle = env.get_vehicle_by_id(route_id['vehicle_id'])
                    print(f"      Added: {vehicle.type} ({vehicle.id})")
        else:
            if DEBUG:
                print(f"    ⚠️  No more vehicles available or couldn't pack remaining orders")
            break  # Can't add more

        iteration += 1

    if DEBUG:
        print(f"\n" + "="*80)
        print(f"🏁 FINAL SOLUTION")
        print("="*80)
        print(f"  Total Routes: {len(solution['routes'])}")
        print(f"  Vehicles Used: {len(used_vehicles)}")
        print(f"  Orders Fulfilled: {len(assigned_orders)}/{len(order_ids)} ({fulfillment_rate*100:.1f}%)")

        if fulfillment_rate < 1.0:
            print(f"\n  ⚠️  UNFULFILLED ORDERS: {len(order_ids) - len(assigned_orders)}")
            unfulfilled_final = set(order_ids) - assigned_orders
            for oid in list(unfulfilled_final)[:5]:  # Show first 5
                order = env.orders[oid]
                print(f"    - {oid}: needs {dict(order.requested_items)}")

        # Final vehicle breakdown by type
        final_vehicle_types = defaultdict(int)
        for route in solution['routes']:
            vehicle = env.get_vehicle_by_id(route['vehicle_id'])
            final_vehicle_types[vehicle.type] += 1

        print(f"\n  Final Vehicle Mix (Total):")
        for vtype, count in sorted(final_vehicle_types.items()):
            print(f"    {vtype}: {count}")
        
        # Detailed breakdown by warehouse
        vehicles_by_warehouse = defaultdict(lambda: {'Light': 0, 'Medium': 0, 'Heavy': 0})
        for route in solution['routes']:
            vehicle = env.get_vehicle_by_id(route['vehicle_id'])
            wh_id = vehicle.home_warehouse_id
            
            if 'Light' in vehicle.type:
                vehicles_by_warehouse[wh_id]['Light'] += 1
            elif 'Medium' in vehicle.type:
                vehicles_by_warehouse[wh_id]['Medium'] += 1
            elif 'Heavy' in vehicle.type:
                vehicles_by_warehouse[wh_id]['Heavy'] += 1
        
        print(f"\n  Vehicle Combination by Warehouse:")
        for wh_id in sorted(vehicles_by_warehouse.keys()):
            counts = vehicles_by_warehouse[wh_id]
            combination = f"{counts['Light']}L_{counts['Medium']}M_{counts['Heavy']}H"
            total = counts['Light'] + counts['Medium'] + counts['Heavy']
            print(f"    {wh_id}: {combination} ({total} vehicles)")

        print("="*80 + "\n")

    return solution


def calculate_cargo_by_warehouse(order_ids, order_data, warehouses):
    """Calculate total cargo volume per warehouse for smart vehicle selection."""
    cargo_by_wh = defaultdict(lambda: {'volume': 0.0, 'weight': 0.0, 'order_count': 0})

    for oid in order_ids:
        wh_id = order_data[oid].get('warehouse_id')
        if wh_id:
            cargo_by_wh[wh_id]['volume'] += order_data[oid]['volume']
            cargo_by_wh[wh_id]['weight'] += order_data[oid]['weight']
            cargo_by_wh[wh_id]['order_count'] += 1

    return dict(cargo_by_wh)


def try_bin_packing_strategy(strategy, wh_orders, wh_vehicles, wh_id,
                              order_data, inventory_remaining, assigned_orders):
    """
    Try a specific bin packing strategy and return results.

    Strategies:
    - 'first_fit': Put item in first bin that fits (fast, good for uniform sizes)
    - 'best_fit': Put item in bin with tightest fit (minimizes waste per bin)
    - 'worst_fit': Put item in bin with most remaining space (spreads load evenly)

    Returns: (bins, assigned_set, inventory_snapshot)
    """
    # Make copies to avoid side effects
    inv_copy = {sku: qty for sku, qty in inventory_remaining[wh_id].items()}
    assigned_copy = set()

    # Sort orders by volume (decreasing)
    wh_orders_sorted = sorted(wh_orders,
                              key=lambda oid: order_data[oid]['volume'],
                              reverse=True)

    bins = []  # (vehicle, weight, volume, orders)

    for oid in wh_orders_sorted:
        req = order_data[oid]
        order = req['order']

        # Skip inventory check - orders were already assigned and reserved in analyze_orders
        # The inventory_remaining passed here already has reservations applied

        # Find bin based on strategy
        bin_idx = -1

        if strategy == 'first_fit':
            # First Fit: Use first bin that fits
            for i, (vehicle, w, v, bin_orders) in enumerate(bins):
                new_w = w + req['weight']
                new_v = v + req['volume']
                if (new_w <= vehicle.capacity_weight * CAPACITY_SAFETY and
                    new_v <= vehicle.capacity_volume * CAPACITY_SAFETY):
                    bin_idx = i
                    break

        elif strategy == 'best_fit':
            # Best Fit: Use bin with minimum remaining space
            best_remaining = float('inf')
            for i, (vehicle, w, v, bin_orders) in enumerate(bins):
                new_w = w + req['weight']
                new_v = v + req['volume']
                if (new_w <= vehicle.capacity_weight * CAPACITY_SAFETY and
                    new_v <= vehicle.capacity_volume * CAPACITY_SAFETY):
                    remaining = (vehicle.capacity_weight - new_w) + (vehicle.capacity_volume - new_v)
                    if remaining < best_remaining:
                        best_remaining = remaining
                        bin_idx = i

        elif strategy == 'worst_fit':
            # Worst Fit: Use bin with maximum remaining space
            worst_remaining = -1.0
            for i, (vehicle, w, v, bin_orders) in enumerate(bins):
                new_w = w + req['weight']
                new_v = v + req['volume']
                if (new_w <= vehicle.capacity_weight * CAPACITY_SAFETY and
                    new_v <= vehicle.capacity_volume * CAPACITY_SAFETY):
                    remaining = (vehicle.capacity_weight - new_w) + (vehicle.capacity_volume - new_v)
                    if remaining > worst_remaining:
                        worst_remaining = remaining
                        bin_idx = i

        if bin_idx >= 0:
            # Pack into selected bin
            vehicle, w, v, bin_orders = bins[bin_idx]
            bins[bin_idx] = (vehicle, w + req['weight'], v + req['volume'], bin_orders + [oid])
            assigned_copy.add(oid)
            # NOTE: Inventory already consumed in analyze_orders, don't consume again

        elif len(bins) < len(wh_vehicles):
            # Create new bin
            vehicle = wh_vehicles[len(bins)]
            if (req['weight'] <= vehicle.capacity_weight * CAPACITY_SAFETY and
                req['volume'] <= vehicle.capacity_volume * CAPACITY_SAFETY):
                bins.append((vehicle, req['weight'], req['volume'], [oid]))
                assigned_copy.add(oid)
                # NOTE: Inventory already consumed in analyze_orders, don't consume again

    return bins, assigned_copy, inv_copy


def pack_into_limited_vehicles(
    env, order_ids, order_data, warehouses, all_vehicles,
    inventory_remaining, adjacency_list, cargo_by_warehouse,
    max_vehicles_per_warehouse=6
):
    """
    TIER 1: SMART VEHICLE SELECTION for cost optimization.

    Strategy:
    - Calculate total cargo volume per warehouse
    - Choose vehicle mix based on volume (prefer LightVans when possible)
    - Volume 0-6m³: Use 2-3 LightVans (£600-900 < £625 MediumTruck)
    - Volume 6-9m³: Use 3-4 LightVans (£900-1200)
    - Volume >9m³: Use HeavyTrucks for efficiency
    - Pack using Best Fit Decreasing for tight utilization
    """
    solution = {"routes": []}
    used_vehicles = set()
    assigned_orders = set()

    # Group orders by warehouse (handle split orders separately)
    orders_by_warehouse = defaultdict(list)
    split_orders = []  # Track orders requiring split delivery
    
    for oid in order_ids:
        order_info = order_data[oid]
        order = order_info['order']

        # Check if this is a split order
        if order_info.get('requires_split'):
            split_orders.append(oid)
            continue

        # Check if marked as unfulfillable
        if order_info.get('unfulfillable'):
            continue  # Skip unfulfillable orders

        # Use warehouse assignment from Phase 1 (don't re-assign)
        wh_id = order_info.get('warehouse_id')
        if wh_id:
            orders_by_warehouse[wh_id].append(oid)

    # Pack each warehouse's orders
    for wh_id, wh_orders in orders_by_warehouse.items():
        warehouse = warehouses[wh_id]

        print(f"  📦 {wh_id}: Packing {len(wh_orders)} orders")

        # ADAPTIVE VEHICLE SELECTION WITH OBJECTIVE FUNCTION
        # Try multiple vehicle mixes and pick the best based on:
        # Score = Fulfilled × 1000 - FixedCost - VehicleCount × 50

        cargo_info = cargo_by_warehouse.get(wh_id, {'volume': 0.0, 'order_count': 0})
        total_volume = cargo_info['volume']

        all_wh_vehicles = [v for v in all_vehicles if v.home_warehouse_id == wh_id]
        light_vans = [v for v in all_wh_vehicles if 'Light' in v.type]
        medium_trucks = [v for v in all_wh_vehicles if 'Medium' in v.type]
        heavy_trucks = [v for v in all_wh_vehicles if 'Heavy' in v.type]

        # BRUTE FORCE: Try ALL possible vehicle combinations
        # Generate every combination of (L light, M medium, H heavy)
        vehicle_mix_strategies = []

        max_light = min(len(light_vans), max_vehicles_per_warehouse)
        max_medium = min(len(medium_trucks), max_vehicles_per_warehouse)
        max_heavy = min(len(heavy_trucks), max_vehicles_per_warehouse)

        for num_light in range(max_light + 1):  # 0 to max_light
            for num_medium in range(max_medium + 1):  # 0 to max_medium
                for num_heavy in range(max_heavy + 1):  # 0 to max_heavy
                    total_vehicles = num_light + num_medium + num_heavy

                    # Skip empty combinations
                    if total_vehicles == 0:
                        continue

                    # Skip combinations exceeding max vehicles
                    if total_vehicles > max_vehicles_per_warehouse:
                        continue

                    # Build vehicle mix
                    mix = []
                    if num_light > 0:
                        mix.extend(light_vans[:num_light])
                    if num_medium > 0:
                        mix.extend(medium_trucks[:num_medium])
                    if num_heavy > 0:
                        mix.extend(heavy_trucks[:num_heavy])

                    strategy_name = f"{num_light}L_{num_medium}M_{num_heavy}H"
                    vehicle_mix_strategies.append((strategy_name, mix))

        if not vehicle_mix_strategies:
            continue

        best_score = (1, float('inf'))  # (doesn't meet target, cost) - both lower is better
        best_candidate = None

        print(f"  🔍 {wh_id}: Testing {len(vehicle_mix_strategies)} vehicle strategies × 3 packing methods...")

        for strategy_name, candidate_vehicles in vehicle_mix_strategies:
            if not candidate_vehicles:
                continue

            # Try ALL 3 packing strategies with this vehicle mix
            for packing_strategy in ['first_fit', 'best_fit', 'worst_fit']:
                # Try packing - returns bins, assigned set, and final inventory state
                trial_bins, trial_assigned, trial_inventory = try_bin_packing_strategy(
                    packing_strategy, wh_orders, candidate_vehicles, wh_id,
                    order_data, inventory_remaining, assigned_orders
                )

                # Calculate metrics
                fulfilled_count = len(trial_assigned)
                fulfillment_pct = (fulfilled_count / len(wh_orders) * 100) if wh_orders else 100
                used_vehicles_in_bins = set(bin[0].id for bin in trial_bins)
                fixed_cost = sum(v.fixed_cost for v in candidate_vehicles if v.id in used_vehicles_in_bins)

                # SIMPLIFIED SCORING: Target 85% fulfillment, then minimize cost
                # Score prioritizes: 1) Meeting 85% target, 2) Lower cost
                meets_target = fulfillment_pct >= (FULFILLMENT_TARGET * 100)
                
                # Sort key: (doesn't meet target, cost)
                # Strategies that meet target (False=0) are ranked above those that don't (True=1)
                score_key = (0 if meets_target else 1, fixed_cost)

                full_name = f"{strategy_name}_{packing_strategy}"

                # Track best by score (lower is better)
                if score_key < best_score:
                    best_score = score_key
                    best_candidate = (fixed_cost, fulfillment_pct, trial_bins, fulfilled_count,
                                    full_name, trial_assigned, trial_inventory, score_key)

        # Use the candidate with LOWEST competition score
        if best_candidate is None:
            print(f"    ⚠️  No valid packing found for {wh_id}")
            continue

        # Unpack: (cost, fulfillment_pct, bins, fulfilled_count, name, assigned_set, inventory, score_key)
        best_cost = best_candidate[0]
        best_fulfillment_pct = best_candidate[1]
        best_bins = best_candidate[2]
        best_fulfilled_count = best_candidate[3]
        best_name = best_candidate[4]
        best_assigned_set = best_candidate[5]
        best_inventory = best_candidate[6]
        best_score_key = best_candidate[7]

        # Use the best result
        bins = best_bins if best_bins else []

        if not bins:
            continue

        # Show selected vehicle combination for this warehouse
        vehicle_counts = {'Light': 0, 'Medium': 0, 'Heavy': 0}
        for vehicle, _, _, bin_orders in bins:
            if 'Light' in vehicle.type:
                vehicle_counts['Light'] += 1
            elif 'Medium' in vehicle.type:
                vehicle_counts['Medium'] += 1
            elif 'Heavy' in vehicle.type:
                vehicle_counts['Heavy'] += 1
        
        combination = f"{vehicle_counts['Light']}L_{vehicle_counts['Medium']}M_{vehicle_counts['Heavy']}H"
        meets_target_str = "✅ Meets 85%" if best_score_key[0] == 0 else "⚠️ Below 85%"
        print(f"    ✅ Selected: {combination} - {best_fulfilled_count}/{len(wh_orders)} orders ({best_fulfillment_pct:.0f}%), Cost=£{best_cost:.0f} [{meets_target_str}]")

        # Update global state with results from winning strategy
        # Add vehicles to used set
        for vehicle, _, _, bin_orders in bins:
            used_vehicles.add(vehicle.id)

        # Update inventory to match the winning strategy's final state
        inventory_remaining[wh_id] = best_inventory

        # Add assigned orders to global set
        assigned_orders.update(best_assigned_set)

        # Build routes
        for vehicle, _, _, bin_orders in bins:
            steps = build_route_with_validation(
                env, vehicle, warehouse, bin_orders, adjacency_list
            )

            if steps:
                solution["routes"].append({
                    "vehicle_id": vehicle.id,
                    "steps": steps
                })

    # OPTIMIZED: Integrate split orders into EXISTING vehicles instead of creating new ones!
    # This saves vehicle fixed costs by reusing vehicles with spare capacity

    if split_orders:
        if not solution["routes"]:
            print(f"\n  ⚠️  No existing routes - skipping split delivery optimization for {len(split_orders)} orders")
        else:
            print(f"\n  🔀 SPLIT DELIVERY OPTIMIZATION: {len(split_orders)} orders to process")
            print(f"  📊 Available routes: {len(solution['routes'])}")
            
            successfully_added = 0
            skipped_capacity = 0
            skipped_rebuild_failed = 0

            # Try to add split orders to existing routes that have capacity
            for split_idx, split_oid in enumerate(split_orders, 1):
                print(f"\n  🔄 [{split_idx}/{len(split_orders)}] Processing {split_oid}...")
                
                split_info = order_data[split_oid]
                split_plan = split_info['split_warehouses']  # [(wh_id, sku_dict), ...]
                
                print(f"    📦 Split Plan ({len(split_plan)} warehouses):")
                for wh_id, sku_needs in split_plan:
                    print(f"      - {wh_id}: {dict(sku_needs)}")

                # NOTE: Skip inventory check - inventory was already reserved in analyze_orders
                # The split_plan was created when inventory was available, and it was immediately consumed
                print(f"    ℹ️  Inventory pre-allocated during order analysis phase")

                # Calculate split order cargo
                split_weight = order_data[split_oid]['weight']
                split_volume = order_data[split_oid]['volume']
                print(f"    📏 Order size: {split_weight:.2f}kg, {split_volume:.2f}m³")

                # Find existing route with spare capacity (prefer closest to destination)
                best_route_idx = None
                best_distance = float('inf')
                split_dest = order_data[split_oid]['destination']
                
                print(f"    🚚 Evaluating {len(solution['routes'])} existing routes...")

                for route_idx, route in enumerate(solution["routes"]):
                    vehicle = env.get_vehicle_by_id(route["vehicle_id"])

                    # Calculate current load
                    current_orders = []
                    for step in route["steps"]:
                        for delivery in step.get('deliveries', []):
                            if delivery['order_id'] not in current_orders:
                                current_orders.append(delivery['order_id'])

                    current_weight = sum(order_data[oid]['weight'] for oid in current_orders if oid in order_data)
                    current_volume = sum(order_data[oid]['volume'] for oid in current_orders if oid in order_data)

                    # Check spare capacity (with small tolerance for floating point precision)
                    spare_weight = vehicle.capacity_weight * CAPACITY_SAFETY - current_weight
                    spare_volume = vehicle.capacity_volume * CAPACITY_SAFETY - current_volume
                    
                    EPSILON = 0.01  # 10g or 0.01m³ tolerance for floating point
                    has_capacity = (spare_weight >= split_weight - EPSILON and 
                                   spare_volume >= split_volume - EPSILON)

                    if has_capacity:
                        # Calculate distance from vehicle's home warehouse to destination
                        home_wh = warehouses[vehicle.home_warehouse_id]
                        distance = dijkstra_distance(adjacency_list, home_wh.location.id, split_dest)
                        
                        status = "✅ CANDIDATE" if distance < best_distance else "  candidate"
                        print(f"      Route {route_idx + 1} ({vehicle.type} @ {vehicle.home_warehouse_id}): "
                              f"spare={spare_weight:.2f}kg/{spare_volume:.3f}m³, dist={distance:.1f}km {status}")
                        
                        if distance < best_distance:
                            best_distance = distance
                            best_route_idx = route_idx
                    else:
                        print(f"      Route {route_idx + 1} ({vehicle.type} @ {vehicle.home_warehouse_id}): "
                              f"❌ NO CAPACITY (spare={spare_weight:.2f}kg/{spare_volume:.3f}m³, need={split_weight:.2f}kg/{split_volume:.3f}m³)")

                if best_route_idx is not None:
                    # Add split order to this route!
                    print(f"    ✅ SELECTED Route {best_route_idx + 1} (distance: {best_distance:.1f}km)")

                    # Rebuild route with split order included
                    existing_route = solution["routes"][best_route_idx]
                    vehicle = env.get_vehicle_by_id(existing_route["vehicle_id"])

                    # Get home warehouse
                    home_wh_id = vehicle.home_warehouse_id
                    home_wh = warehouses[home_wh_id]

                    # Extract current orders from existing route
                    route_orders = []
                    for step in existing_route["steps"]:
                        for delivery in step.get('deliveries', []):
                            if delivery['order_id'] not in route_orders:
                                route_orders.append(delivery['order_id'])

                    print(f"    🔨 Rebuilding route with {len(route_orders)} existing orders + split order...")

                    # Build new route with split order
                    new_steps = build_route_with_split_delivery(
                        env, vehicle, home_wh, route_orders, split_oid, split_plan,
                        warehouses, adjacency_list
                    )

                    if new_steps:
                        print(f"    ✅ SUCCESS: Route rebuilt with {len(new_steps)} steps")
                        solution["routes"][best_route_idx]["steps"] = new_steps
                        assigned_orders.add(split_oid)
                        successfully_added += 1
                        
                        # NOTE: Don't consume inventory here - it was already consumed in analyze_orders
                    else:
                        print(f"    ❌ FAILED: Route rebuild returned empty")
                        skipped_rebuild_failed += 1
                else:
                    print(f"    ❌ SKIPPED: No route has sufficient capacity")
                    skipped_capacity += 1
            
            print(f"\n  📊 Split Delivery Results:")
            print(f"    ✅ Successfully added: {successfully_added}/{len(split_orders)}")
            if skipped_capacity > 0:
                print(f"    ⚠️  Skipped (no capacity): {skipped_capacity}")
            if skipped_rebuild_failed > 0:
                print(f"    ⚠️  Skipped (rebuild failed): {skipped_rebuild_failed}")

    return solution, used_vehicles, assigned_orders


def pack_fallback_orders(
    env, unfulfilled_orders, order_data, warehouses, all_vehicles,
    used_vehicles, inventory_remaining, adjacency_list
):
    """
    TIER 2: Pack unfulfilled orders into additional vehicles.
    More aggressive - packs multiple orders per vehicle.
    """
    fallback_routes = []
    fallback_used = set()
    newly_assigned = set()

    # Group unfulfilled by warehouse
    unfulfilled_by_wh = defaultdict(list)
    for oid in unfulfilled_orders:
        order = order_data[oid]['order']

        for wh_id, inv in inventory_remaining.items():
            has_all = True
            for sku_id, qty in order.requested_items.items():
                if inv.get(sku_id, 0) < qty:
                    has_all = False
                    break

            if has_all:
                unfulfilled_by_wh[wh_id].append(oid)
                break

    # Pack each warehouse's unfulfilled orders
    for wh_id, wh_unfulfilled in unfulfilled_by_wh.items():
        warehouse = warehouses[wh_id]

        # Calculate unfulfilled cargo volume
        unfulfilled_volume = sum(order_data[oid]['volume'] for oid in wh_unfulfilled)

        # Get unused vehicles
        available = [v for v in all_vehicles
                    if v.home_warehouse_id == wh_id and v.id not in used_vehicles]

        if PREFER_SMALL_VEHICLES and unfulfilled_volume <= 6.0:
            # COST OPTIMIZATION: For small unfulfilled cargo, prefer LightVans
            light_vans = [v for v in available if 'Light' in v.type]
            others = [v for v in available if 'Light' not in v.type]
            others.sort(key=lambda v: (v.capacity_weight, v.capacity_volume))
            available = light_vans + others  # LightVans first
        else:
            # Large unfulfilled or no preference: use largest for capacity
            available.sort(key=lambda v: (v.capacity_weight, v.capacity_volume), reverse=True)

        if not available:
            continue

        # Pack into bins (use as many vehicles as needed, up to MAX_FALLBACK_VEHICLES_PER_WH per warehouse per iteration)
        bins = []  # (vehicle, weight, volume, orders)

        for oid in wh_unfulfilled:
            if len(bins) >= MAX_FALLBACK_VEHICLES_PER_WH:  # Limit per iteration to prevent explosion
                break

            req = order_data[oid]
            order = req['order']

            # Check inventory
            can_fulfill = True
            for sku_id, qty in order.requested_items.items():
                if inventory_remaining[wh_id].get(sku_id, 0) < qty:
                    can_fulfill = False
                    break

            if not can_fulfill:
                continue

            # Try to fit in existing fallback bin
            packed = False
            for i, (vehicle, w, v, bin_orders) in enumerate(bins):
                new_w = w + req['weight']
                new_v = v + req['volume']

                if (new_w <= vehicle.capacity_weight * CAPACITY_SAFETY and
                    new_v <= vehicle.capacity_volume * CAPACITY_SAFETY):
                    bins[i] = (vehicle, new_w, new_v, bin_orders + [oid])
                    packed = True

                    # Reserve inventory
                    for sku_id, qty in order.requested_items.items():
                        inventory_remaining[wh_id][sku_id] -= qty
                    newly_assigned.add(oid)
                    break

            # Need new fallback vehicle
            if not packed and len(bins) < 3:
                for vehicle in available:
                    if vehicle.id not in used_vehicles:
                        if (req['weight'] <= vehicle.capacity_weight * CAPACITY_SAFETY and
                            req['volume'] <= vehicle.capacity_volume * CAPACITY_SAFETY):
                            bins.append((vehicle, req['weight'], req['volume'], [oid]))
                            used_vehicles.add(vehicle.id)
                            fallback_used.add(vehicle.id)

                            # Reserve inventory
                            for sku_id, qty in order.requested_items.items():
                                inventory_remaining[wh_id][sku_id] -= qty
                            newly_assigned.add(oid)
                            break

        # Build routes for fallback bins
        for vehicle, _, _, bin_orders in bins:
            steps = build_route_with_validation(
                env, vehicle, warehouse, bin_orders, adjacency_list
            )

            if steps:
                fallback_routes.append({
                    "vehicle_id": vehicle.id,
                    "steps": steps
                })

    return fallback_routes, fallback_used, newly_assigned


def build_adjacency_list(road_network: Dict) -> Dict:
    """Build adjacency list with distances from edges."""
    adjacency_list = defaultdict(list)
    edges = road_network.get('edges', [])

    for edge in edges:
        from_node = edge['from']
        to_node = edge['to']
        distance = edge['distance']
        adjacency_list[from_node].append((to_node, distance))

    return dict(adjacency_list)


def dijkstra_distance(adjacency_list: Dict, start: int, end: int) -> float:
    """Calculate shortest path distance using Dijkstra."""
    if start == end:
        return 0.0

    if start not in adjacency_list:
        return float('inf')

    distances = {start: 0.0}
    pq = [(0.0, start)]
    visited = set()

    while pq:
        curr_dist, curr_node = heapq.heappop(pq)

        if curr_node in visited:
            continue

        visited.add(curr_node)

        if curr_node == end:
            return curr_dist

        if curr_node not in adjacency_list:
            continue

        for neighbor, edge_dist in adjacency_list[curr_node]:
            if neighbor not in visited:
                new_dist = curr_dist + edge_dist
                if neighbor not in distances or new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    heapq.heappush(pq, (new_dist, neighbor))

    return float('inf')


def dijkstra_path(adjacency_list: Dict, start: int, end: int) -> List[int]:
    """Find shortest path using Dijkstra, returns nodes excluding start."""
    if start == end:
        return []

    if start not in adjacency_list:
        return []

    distances = {start: 0.0}
    previous = {start: None}
    pq = [(0.0, start)]
    visited = set()

    while pq:
        curr_dist, curr_node = heapq.heappop(pq)

        if curr_node in visited:
            continue

        visited.add(curr_node)

        if curr_node == end:
            path = []
            node = end
            while node is not None and node != start:
                path.append(node)
                node = previous.get(node)
            path.reverse()
            return path

        if curr_node not in adjacency_list:
            continue

        for neighbor, edge_dist in adjacency_list[curr_node]:
            if neighbor not in visited:
                new_dist = curr_dist + edge_dist
                if neighbor not in distances or new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = curr_node
                    heapq.heappush(pq, (new_dist, neighbor))

    return []


def analyze_orders(env, order_ids: List[str], warehouses: Dict, adjacency_list: Dict) -> tuple:
    """
    SIMPLE random warehouse assignment - like solver_29.
    Assigns each order to first available warehouse with inventory.
    Keeps inventory reservation and split delivery features.
    
    Returns: (order_data, reserved_inventory)
    """
    order_data = {}

    # Track reserved inventory during assignment
    reserved_inventory = {wh_id: dict(warehouse.inventory) for wh_id, warehouse in warehouses.items()}

    # SIMPLE: Process orders in random order (no distance calculations)
    for order_id in order_ids:
        order = env.orders[order_id]
        dest_node = order.destination.id

        # Calculate weight and volume
        total_weight = 0.0
        total_volume = 0.0
        for sku_id, qty in order.requested_items.items():
            sku = env.skus[sku_id]
            total_weight += sku.weight * qty
            total_volume += sku.volume * qty

        # Find ANY warehouse with full inventory (random order - whatever dict gives us)
        best_wh = None
        for wh_id in warehouses.keys():
            if all(reserved_inventory[wh_id].get(sku_id, 0) >= qty for sku_id, qty in order.requested_items.items()):
                best_wh = wh_id
                # Reserve inventory
                for sku_id, qty in order.requested_items.items():
                    reserved_inventory[wh_id][sku_id] -= qty
                break

        # Try split delivery if no single warehouse works
        if best_wh is None:
            # Check if order can be fulfilled by combining warehouses (using reserved inventory!)
            split_warehouses = attempt_split_delivery(order, reserved_inventory, warehouses, adjacency_list, dest_node)
            
            if split_warehouses:
                # Reserve inventory from all contributing warehouses
                for wh_id, sku_dict in split_warehouses:
                    for sku_id, qty in sku_dict.items():
                        reserved_inventory[wh_id][sku_id] -= qty

                order_data[order_id] = {
                    'weight': total_weight,
                    'volume': total_volume,
                    'warehouse_id': None,
                    'split_warehouses': split_warehouses,
                    'destination': dest_node,
                    'order': order,
                    'requires_split': True
                }
            else:
                # Unfulfillable - assign to first warehouse (NO inventory consumption)
                first_wh = list(warehouses.keys())[0]
                order_data[order_id] = {
                    'weight': total_weight,
                    'volume': total_volume,
                    'warehouse_id': first_wh,
                    'destination': dest_node,
                    'order': order,
                    'unfulfillable': True
                }
        else:
            order_data[order_id] = {
                'weight': total_weight,
                'volume': total_volume,
                'warehouse_id': best_wh,
                'destination': dest_node,
                'order': order
            }

    return order_data, reserved_inventory


def attempt_split_delivery(order, reserved_inventory: Dict, warehouses: Dict, adjacency_list: Dict, dest_node: int) -> List[Tuple[str, Dict]]:
    """
    Attempt to fulfill an order by splitting it across multiple warehouses.

    Args:
        order: The order to fulfill
        reserved_inventory: Current reserved inventory state (not static warehouse inventory!)
        warehouses: Warehouse objects
        adjacency_list: Road network
        dest_node: Destination node

    Returns:
        List of (warehouse_id, sku_dict) tuples if order can be fulfilled
        Empty list if order cannot be fulfilled even with split delivery
    """
    # Get all SKUs needed for this order
    required_skus = dict(order.requested_items)
    split_plan = []
    used_warehouses = set()  # Track which warehouses we've already used

    # Try to fulfill each SKU from available warehouses
    remaining_skus = dict(required_skus)

    while remaining_skus:
        # Evaluate all warehouses for their contribution to remaining SKUs
        candidates = []
        
        for wh_id, warehouse in warehouses.items():
            # Skip warehouses already used in this split
            if wh_id in used_warehouses:
                continue
                
            contribution = {}
            
            for sku_id, qty_needed in remaining_skus.items():
                # Use reserved_inventory, not warehouse.inventory!
                available = reserved_inventory[wh_id].get(sku_id, 0)
                if available > 0:
                    contribution[sku_id] = min(available, qty_needed)
            
            if contribution:
                distance = dijkstra_distance(adjacency_list, warehouse.location.id, dest_node)
                total_qty = sum(contribution.values())
                # Sort by: DISTANCE FIRST (closest), then more SKUs, then more quantity
                candidates.append((distance, -len(contribution), -total_qty, wh_id, contribution))
        
        if not candidates:
            # No warehouse can fulfill remaining SKUs - order cannot be completed
            return []
        
        # Pick the best warehouse (closest with available SKUs)
        _, _, _, best_wh_id, best_contribution = min(candidates)
        split_plan.append((best_wh_id, best_contribution))
        used_warehouses.add(best_wh_id)  # Mark this warehouse as used
        
        # Update remaining SKUs
        for sku_id, qty_fulfilled in best_contribution.items():
            remaining_skus[sku_id] -= qty_fulfilled
            if remaining_skus[sku_id] <= 0:
                del remaining_skus[sku_id]
        
        # Safety: Limit to 3 warehouses maximum (complexity management)
        if len(split_plan) >= 3:
            break
    
    # Check if all SKUs are fulfilled
    if remaining_skus:
        return []  # Still have unfulfilled SKUs
    
    return split_plan


def cluster_orders_by_location(env, order_data: Dict, max_clusters: int = 1) -> List[List[str]]:
    """
    Cluster orders by geographic proximity using K-means-like approach.
    Goal: Group nearby orders to minimize vehicles.
    """
    if len(order_data) <= max_clusters:
        # Few orders: one cluster per order or combine all
        return [[oid] for oid in order_data.keys()]

    # Extract order locations
    order_locations = {}
    for oid, data in order_data.items():
        dest_node = data['destination']
        node = env.nodes[dest_node]
        order_locations[oid] = (node.lat, node.lon)

    # Simple K-means clustering
    clusters = kmeans_cluster(order_locations, k=max_clusters)

    # Merge small clusters to maximize truck utilization
    clusters = merge_small_clusters(clusters, order_data, min_size=len(order_data) // max_clusters)

    return clusters


def kmeans_cluster(locations: Dict[str, Tuple[float, float]], k: int, max_iter: int = 10) -> List[List[str]]:
    """
    Improved K-means clustering on lat/lon coordinates.

    MERGED OPTIMIZATIONS:
    1. Better initialization: Pick spread-out centroids (k-means++)
    2. Use Haversine distance for geographic accuracy
    3. Limit iterations to prevent slow convergence
    """
    order_ids = list(locations.keys())

    if len(order_ids) <= k:
        return [[oid] for oid in order_ids]

    # K-means++ initialization: Pick well-separated initial centroids
    centroids = []
    centroid_ids = []

    # First centroid: random
    first_id = order_ids[0]
    centroid_ids.append(first_id)
    centroids.append(locations[first_id])

    # Subsequent centroids: Pick points far from existing centroids
    for _ in range(min(k - 1, len(order_ids) - 1)):
        max_min_dist = -1
        farthest_id = None

        for oid in order_ids:
            if oid in centroid_ids:
                continue

            # Find minimum distance to existing centroids
            min_dist_to_centroid = min(
                haversine_distance(locations[oid], c) for c in centroids
            )

            if min_dist_to_centroid > max_min_dist:
                max_min_dist = min_dist_to_centroid
                farthest_id = oid

        if farthest_id:
            centroid_ids.append(farthest_id)
            centroids.append(locations[farthest_id])

    k = len(centroids)

    # K-means iterations
    for _ in range(max_iter):
        # Assign to nearest centroid using Haversine distance
        clusters = [[] for _ in range(k)]
        for oid in order_ids:
            loc = locations[oid]
            nearest_idx = min(range(k), key=lambda i: haversine_distance(loc, centroids[i]))
            clusters[nearest_idx].append(oid)

        # Remove empty clusters
        clusters = [c for c in clusters if c]

        # Update centroids (geographic center)
        new_centroids = []
        for cluster in clusters:
            lats = [locations[oid][0] for oid in cluster]
            lons = [locations[oid][1] for oid in cluster]
            new_centroids.append((sum(lats) / len(lats), sum(lons) / len(lons)))

        # Check convergence (centroids barely moved)
        if len(new_centroids) == len(centroids):
            converged = True
            for old_c, new_c in zip(centroids, new_centroids):
                if haversine_distance(old_c, new_c) > 0.001:  # ~100m threshold
                    converged = False
                    break
            if converged:
                break

        centroids = new_centroids
        k = len(centroids)

    return [c for c in clusters if c]


def haversine_distance(loc1: Tuple[float, float], loc2: Tuple[float, float]) -> float:
    """
    Calculate Haversine distance between two lat/lon points.
    Returns distance in kilometers (more accurate for geographic coordinates).
    """
    lat1, lon1 = loc1
    lat2, lon2 = loc2

    # Earth radius in km
    R = 6371.0

    # Convert to radians
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    # Haversine formula
    a = (math.sin(delta_lat / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) *
         math.sin(delta_lon / 2) ** 2)

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance


def euclidean_dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Euclidean distance between two points (kept for backward compatibility)."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def merge_small_clusters(clusters: List[List[str]], order_data: Dict, min_size: int) -> List[List[str]]:
    """Merge clusters that are too small to maximize vehicle utilization."""
    if len(clusters) <= 1:
        return clusters

    # Sort clusters by total weight (proxy for size)
    def cluster_weight(cluster):
        return sum(order_data[oid]['weight'] for oid in cluster)

    clusters.sort(key=cluster_weight, reverse=True)

    # Merge small clusters into larger ones
    merged = []
    small_orders = []

    for cluster in clusters:
        if len(cluster) >= min_size or len(merged) == 0:
            merged.append(cluster)
        else:
            small_orders.extend(cluster)

    # Add small orders to first cluster
    if small_orders and merged:
        merged[0].extend(small_orders)
    elif small_orders:
        merged.append(small_orders)

    return merged


def pack_orders_minimum_vehicles(
    env,
    order_data: Dict,
    order_ids: List[str],
    warehouses: Dict,
    all_vehicles: List
) -> Dict[str, List[str]]:
    """
    Pack all orders into absolute minimum number of vehicles.
    Try to achieve 1-3 vehicles total.
    """
    # Group vehicles by warehouse, sort by capacity (largest first for max packing)
    vehicles_by_wh = defaultdict(list)
    for vehicle in all_vehicles:
        vehicles_by_wh[vehicle.home_warehouse_id].append(vehicle)

    for wh_id in vehicles_by_wh:
        # Sort by capacity: largest first to maximize orders per vehicle
        # Prioritize HeavyTruck > MediumTruck > LightVan
        def vehicle_priority(v):
            if 'Heavy' in v.type:
                return (3, v.capacity_weight, v.capacity_volume)
            elif 'Medium' in v.type:
                return (2, v.capacity_weight, v.capacity_volume)
            else:
                return (1, v.capacity_weight, v.capacity_volume)

        vehicles_by_wh[wh_id].sort(key=vehicle_priority, reverse=True)

    # Group orders by warehouse
    orders_by_wh = defaultdict(list)
    for oid in order_ids:
        wh_id = order_data[oid]['warehouse_id']
        orders_by_wh[wh_id].append(oid)

    vehicle_assignments = {}
    used_vehicles = set()

    # Track inventory
    inventory_remaining = {}
    for wh_id, warehouse in warehouses.items():
        inventory_remaining[wh_id] = dict(warehouse.inventory)

    # For each warehouse, pack orders into minimum vehicles
    for wh_id, wh_orders in orders_by_wh.items():
        # Sort orders by weight (heaviest first for better bin packing)
        wh_orders_sorted = sorted(
            wh_orders,
            key=lambda oid: (order_data[oid]['weight'], order_data[oid]['volume']),
            reverse=True
        )

        available_vehicles = vehicles_by_wh[wh_id]
        bins = []  # (vehicle, weight, volume, orders)

        for oid in wh_orders_sorted:
            req = order_data[oid]

            # Check inventory
            order = req['order']
            can_fulfill = True
            for sku_id, qty in order.requested_items.items():
                if inventory_remaining[wh_id].get(sku_id, 0) < qty:
                    can_fulfill = False
                    break

            if not can_fulfill:
                continue

            # Try to fit in existing bin (First Fit Decreasing)
            packed = False
            for i, (vehicle, w, v, bin_orders) in enumerate(bins):
                new_w = w + req['weight']
                new_v = v + req['volume']

                # BULLETPROOF: Check BOTH capacity AND inventory BEFORE packing
                if (new_w <= vehicle.capacity_weight * CAPACITY_SAFETY and
                    new_v <= vehicle.capacity_volume * CAPACITY_SAFETY):

                    # RESERVE inventory immediately
                    for sku_id, qty in order.requested_items.items():
                        inventory_remaining[wh_id][sku_id] -= qty

                    bins[i] = (vehicle, new_w, new_v, bin_orders + [oid])
                    packed = True
                    break

            # Need new vehicle
            if not packed:
                # Get largest unused vehicle to maximize capacity and minimize vehicle count
                assigned_vehicle = None
                for vehicle in available_vehicles:
                    if vehicle.id not in used_vehicles:
                        if (req['weight'] <= vehicle.capacity_weight * CAPACITY_SAFETY and
                            req['volume'] <= vehicle.capacity_volume * CAPACITY_SAFETY):
                            assigned_vehicle = vehicle
                            break

                if assigned_vehicle:
                    # RESERVE inventory immediately
                    for sku_id, qty in order.requested_items.items():
                        inventory_remaining[wh_id][sku_id] -= qty

                    bins.append((assigned_vehicle, req['weight'], req['volume'], [oid]))
                    used_vehicles.add(assigned_vehicle.id)

        # Record assignments
        for vehicle, _, _, bin_orders in bins:
            vehicle_assignments[vehicle.id] = bin_orders

    return vehicle_assignments


def assign_vehicles_with_backtracking(
    env,
    order_data: Dict,
    clusters: List[List[str]],
    warehouses: Dict,
    all_vehicles: List
) -> Dict[str, List[str]]:
    """
    Assign vehicles with backtracking: try smallest first, backtrack if doesn't fit.
    Goal: Minimize vehicle count and prefer smaller vehicles.
    Track inventory per warehouse to avoid conflicts.
    """
    # Group vehicles by warehouse and sort by capacity (smallest first)
    vehicles_by_wh = defaultdict(list)
    for vehicle in all_vehicles:
        vehicles_by_wh[vehicle.home_warehouse_id].append(vehicle)

    for wh_id in vehicles_by_wh:
        vehicles_by_wh[wh_id].sort(key=lambda v: (v.capacity_weight, v.capacity_volume))

    vehicle_assignments = {}
    used_vehicles = set()

    # Track inventory consumption per warehouse
    inventory_remaining = {}
    for wh_id, warehouse in warehouses.items():
        inventory_remaining[wh_id] = dict(warehouse.inventory)

    # For each cluster, try to fit in smallest possible vehicle
    for cluster in clusters:
        if not cluster:
            continue

        # Determine primary warehouse for this cluster
        wh_counts = defaultdict(int)
        for oid in cluster:
            wh_counts[order_data[oid]['warehouse_id']] += 1
        primary_wh = max(wh_counts.items(), key=lambda x: x[1])[0]

        # Check if cluster has sufficient inventory
        # IMPORTANT: filter_orders_by_inventory RESERVES inventory as it validates!
        cluster_valid, cluster_filtered = filter_orders_by_inventory(
            cluster, order_data, env, inventory_remaining[primary_wh]
        )

        if not cluster_filtered:
            continue

        # Try to fit all cluster orders in one vehicle (backtracking)
        vehicle_id = try_fit_in_vehicle(
            cluster_filtered, order_data, vehicles_by_wh[primary_wh], used_vehicles
        )

        if vehicle_id:
            vehicle_assignments[vehicle_id] = cluster_filtered
            used_vehicles.add(vehicle_id)
            # NO update_inventory call - already reserved in filter_orders_by_inventory!
        else:
            # Split cluster and try again
            split_assignments = split_and_assign(
                cluster_filtered, order_data, vehicles_by_wh[primary_wh], used_vehicles
            )
            vehicle_assignments.update(split_assignments)
            # NO update_inventory call - already reserved in filter_orders_by_inventory!

    return vehicle_assignments


def filter_orders_by_inventory(orders: List[str], order_data: Dict, env, inventory: Dict) -> Tuple[bool, List[str]]:
    """
    Filter orders to only those that can be fulfilled with available inventory.

    CRITICAL CHANGE: This function NO LONGER RESERVES inventory.
    It only checks availability. Inventory is reserved when routes are actually built.
    """
    valid_orders = []

    for oid in orders:
        order = order_data[oid]['order']
        can_fulfill = True

        # Check if ALL items for this order are available
        for sku_id, qty in order.requested_items.items():
            if inventory.get(sku_id, 0) < qty:
                can_fulfill = False
                break

        if can_fulfill:
            valid_orders.append(oid)
            # DO NOT RESERVE HERE - let routes be built first

    return (len(valid_orders) == len(orders), valid_orders)


def update_inventory(orders: List[str], order_data: Dict, env, inventory: Dict):
    """Update inventory after assigning orders to a vehicle."""
    for oid in orders:
        order = order_data[oid]['order']
        for sku_id, qty in order.requested_items.items():
            inventory[sku_id] = inventory.get(sku_id, 0) - qty


def try_fit_in_vehicle(
    orders: List[str],
    order_data: Dict,
    available_vehicles: List,
    used_vehicles: Set[str]
) -> str:
    """Try to fit orders in smallest available vehicle (backtracking)."""
    total_weight = sum(order_data[oid]['weight'] for oid in orders)
    total_volume = sum(order_data[oid]['volume'] for oid in orders)

    # If many orders, prioritize larger vehicles for efficiency
    # This helps reduce vehicle count
    if len(orders) > 15:
        # Try largest unused vehicle first for big clusters
        for vehicle in reversed(available_vehicles):
            if vehicle.id in used_vehicles:
                continue

            if (total_weight <= vehicle.capacity_weight * CAPACITY_SAFETY and
                total_volume <= vehicle.capacity_volume * CAPACITY_SAFETY):
                return vehicle.id
    else:
        # For smaller clusters, prefer smaller vehicles (lower fixed cost)
        for vehicle in available_vehicles:
            if vehicle.id in used_vehicles:
                continue

            # Check capacity with safety factor
            if (total_weight <= vehicle.capacity_weight * CAPACITY_SAFETY and
                total_volume <= vehicle.capacity_volume * CAPACITY_SAFETY):
                return vehicle.id

    return None


def split_and_assign(
    orders: List[str],
    order_data: Dict,
    available_vehicles: List,
    used_vehicles: Set[str]
) -> Dict[str, List[str]]:
    """
    Split orders and assign to multiple vehicles using bin packing.
    NOTE: This function assumes orders have already been filtered for inventory availability.
    The caller (assign_vehicles_with_backtracking) handles inventory updates after this returns.
    """
    assignments = {}

    # Sort orders by size (largest first) for better packing
    orders_sorted = sorted(
        orders,
        key=lambda oid: (order_data[oid]['weight'], order_data[oid]['volume']),
        reverse=True
    )

    bins = []  # (vehicle, weight, volume, orders)

    for oid in orders_sorted:
        req = order_data[oid]
        packed = False

        # Try existing bins (First Fit Decreasing)
        for i, (vehicle, w, v, bin_orders) in enumerate(bins):
            new_w = w + req['weight']
            new_v = v + req['volume']

            if (new_w <= vehicle.capacity_weight * CAPACITY_SAFETY and
                new_v <= vehicle.capacity_volume * CAPACITY_SAFETY):
                bins[i] = (vehicle, new_w, new_v, bin_orders + [oid])
                packed = True
                break

        # Need new vehicle
        if not packed:
            vehicle_id = try_fit_in_vehicle([oid], order_data, available_vehicles, used_vehicles)
            if vehicle_id:
                vehicle = next(v for v in available_vehicles if v.id == vehicle_id)
                bins.append((vehicle, req['weight'], req['volume'], [oid]))
                used_vehicles.add(vehicle_id)

    # Convert bins to assignments
    for vehicle, _, _, bin_orders in bins:
        assignments[vehicle.id] = bin_orders

    return assignments
#imp to check
def build_split_delivery_route(
    env, order_id: str, split_plan: List[Tuple[str, Dict]],
    warehouses: Dict, all_vehicles: List, used_vehicles: Set,
    inventory_remaining: Dict, adjacency_list: Dict
) -> Dict:
    """
    Build a route that picks up from multiple warehouses to fulfill one order.
    
    Args:
        order_id: The order requiring split delivery
        split_plan: List of (warehouse_id, sku_dict) tuples
        
    Returns:
        Route dict with vehicle_id and steps
    """
    order = env.orders[order_id]
    dest_node = order.destination.id
    
    # Calculate total cargo (VALIDATE BEFORE BUILDING ROUTE)
    total_weight = 0.0
    total_volume = 0.0
    for sku_id, qty in order.requested_items.items():
        sku = env.skus[sku_id]
        total_weight += sku.weight * qty
        total_volume += sku.volume * qty

    # Find warehouse with most SKUs in split plan to determine home base
    # (optimization: start from warehouse contributing most)
    first_wh_id = max(split_plan, key=lambda x: len(x[1]))[0]
    first_warehouse = warehouses[first_wh_id]

    # Find suitable vehicle from first warehouse (check capacity BEFORE route building)
    available_vehicles = [v for v in all_vehicles
                         if v.home_warehouse_id == first_wh_id
                         and v.id not in used_vehicles
                         and total_weight <= v.capacity_weight * CAPACITY_SAFETY
                         and total_volume <= v.capacity_volume * CAPACITY_SAFETY]
    
    if not available_vehicles:
        return None
    
    # Use largest available vehicle for split delivery (more complexity = need buffer)
    vehicle = max(available_vehicles, key=lambda v: (v.capacity_weight, v.capacity_volume))
    used_vehicles.add(vehicle.id)
    
    # Build route: Home → WH1 (pickup) → WH2 (pickup) → ... → Delivery → Home
    steps = []
    current_node = first_warehouse.location.id

    # OPTIMIZATION: Use nearest-neighbor to order warehouse visits optimally
    warehouse_nodes = {}
    for wh_id, sku_needs in split_plan:
        wh_node = warehouses[wh_id].location.id
        warehouse_nodes[wh_node] = (wh_id, sku_needs)

    # Get optimal tour for warehouse pickups using TSP heuristic
    warehouse_tour = nearest_neighbor_tour(current_node, list(warehouse_nodes.keys()), adjacency_list)

    # Visit each warehouse in OPTIMIZED order (nearest-neighbor TSP)
    for wh_node in warehouse_tour:
        wh_id, sku_needs = warehouse_nodes[wh_node]

        # Navigate to this warehouse (if not already there)
        if current_node != wh_node:
            path = dijkstra_path(adjacency_list, current_node, wh_node)
            if not path:
                return None  # Cannot reach warehouse

            # Add intermediate nodes
            for intermediate in path[:-1]:
                steps.append({
                    'node_id': intermediate,
                    'pickups': [],
                    'deliveries': [],
                    'unloads': []
                })
            current_node = wh_node
        
        # Pickup at this warehouse
        pickups = []
        for sku_id, qty in sku_needs.items():
            pickups.append({
                'warehouse_id': wh_id,
                'sku_id': sku_id,
                'quantity': qty
            })
        
        steps.append({
            'node_id': wh_node,
            'pickups': pickups,
            'deliveries': [],
            'unloads': []
        })
    
    # Navigate to delivery destination
    path_to_dest = dijkstra_path(adjacency_list, current_node, dest_node)
    if not path_to_dest:
        return None
    
    for intermediate in path_to_dest[:-1]:
        steps.append({
            'node_id': intermediate,
            'pickups': [],
            'deliveries': [],
            'unloads': []
        })
    
    # Deliver all items at destination
    deliveries = []
    for sku_id, qty in order.requested_items.items():
        deliveries.append({
            'order_id': order_id,
            'sku_id': sku_id,
            'quantity': qty
        })
    
    steps.append({
        'node_id': dest_node,
        'pickups': [],
        'deliveries': deliveries,
        'unloads': []
    })
    
    # Return to home warehouse
    home_node = first_warehouse.location.id
    return_path = dijkstra_path(adjacency_list, dest_node, home_node)
    
    if return_path:
        for node in return_path[:-1]:
            steps.append({
                'node_id': node,
                'pickups': [],
                'deliveries': [],
                'unloads': []
            })
        
        steps.append({
            'node_id': home_node,
            'pickups': [],
            'deliveries': [],
            'unloads': []
        })
    
    return {
        'vehicle_id': vehicle.id,
        'steps': steps
    }


def build_route_with_split_delivery(
    env,
    vehicle,
    home_warehouse,
    normal_orders: List[str],
    split_order_id: str,
    split_plan: List,  # [(wh_id, sku_dict), ...]
    warehouses: Dict,
    adjacency_list: Dict
) -> List[Dict]:
    """
    Build route that visits MULTIPLE warehouses for split delivery.

    Route flow:
    1. Start at home warehouse (WH-A)
    2. Pick up items for normal orders from WH-A
    3. Pick up split order items from WH-A (if any)
    4. Deliver normal orders
    5. Go to OTHER warehouse (WH-B)
    6. Pick up remaining split order items from WH-B
    7. Deliver split order
    8. Return to home warehouse (WH-A)
    """
    home_node = home_warehouse.location.id
    steps = []

    # Step 1: Pickup from HOME warehouse
    pickups_from_home = []
    total_weight = 0.0
    total_volume = 0.0

    # Add normal order pickups
    for oid in normal_orders:
        order = env.orders[oid]
        for sku_id, qty in order.requested_items.items():
            sku = env.skus[sku_id]
            pickups_from_home.append({
                'warehouse_id': home_warehouse.id,
                'sku_id': sku_id,
                'quantity': qty
            })
            total_weight += sku.weight * qty
            total_volume += sku.volume * qty

    # Add split order items from HOME warehouse
    split_order = env.orders[split_order_id]
    for wh_id, sku_dict in split_plan:
        if wh_id == home_warehouse.id:
            for sku_id, qty in sku_dict.items():
                sku = env.skus[sku_id]
                pickups_from_home.append({
                    'warehouse_id': home_warehouse.id,
                    'sku_id': sku_id,
                    'quantity': qty
                })
                total_weight += sku.weight * qty
                total_volume += sku.volume * qty

    # Capacity check
    if (total_weight > vehicle.capacity_weight * CAPACITY_SAFETY or
        total_volume > vehicle.capacity_volume * CAPACITY_SAFETY):
        return []  # Exceeds capacity

    steps.append({
        'node_id': home_node,
        'pickups': pickups_from_home,
        'deliveries': [],
        'unloads': []
    })

    # Step 2: Deliver normal orders using nearest neighbor
    if normal_orders:
        delivery_nodes = {}
        for oid in normal_orders:
            order = env.orders[oid]
            dest = order.destination.id
            if dest not in delivery_nodes:
                delivery_nodes[dest] = []
            delivery_nodes[dest].append(oid)

        tour = nearest_neighbor_tour(home_node, list(delivery_nodes.keys()), adjacency_list)

        prev_node = home_node
        for delivery_node in tour:
            path = dijkstra_path(adjacency_list, prev_node, delivery_node)
            if not path:
                continue

            # Add intermediate nodes
            for intermediate in path[:-1]:
                steps.append({
                    'node_id': intermediate,
                    'pickups': [],
                    'deliveries': [],
                    'unloads': []
                })

            # Delivery step
            deliveries = []
            for oid in delivery_nodes[delivery_node]:
                order = env.orders[oid]
                for sku_id, qty in order.requested_items.items():
                    deliveries.append({
                        'order_id': oid,
                        'sku_id': sku_id,
                        'quantity': qty
                    })

            steps.append({
                'node_id': delivery_node,
                'pickups': [],
                'deliveries': deliveries,
                'unloads': []
            })

            prev_node = delivery_node
    else:
        prev_node = home_node

    # Step 3: Visit OTHER warehouse for split order pickup
    other_wh_id = None
    other_wh_pickups = []

    for wh_id, sku_dict in split_plan:
        if wh_id != home_warehouse.id:
            other_wh_id = wh_id
            other_wh = warehouses[wh_id]

            for sku_id, qty in sku_dict.items():
                other_wh_pickups.append({
                    'warehouse_id': wh_id,
                    'sku_id': sku_id,
                    'quantity': qty
                })

    if other_wh_id:
        other_wh = warehouses[other_wh_id]
        other_wh_node = other_wh.location.id

        # Path to other warehouse
        path_to_other = dijkstra_path(adjacency_list, prev_node, other_wh_node)

        if path_to_other:
            # Intermediate nodes
            for intermediate in path_to_other[:-1]:
                steps.append({
                    'node_id': intermediate,
                    'pickups': [],
                    'deliveries': [],
                    'unloads': []
                })

            # Pickup at other warehouse
            steps.append({
                'node_id': other_wh_node,
                'pickups': other_wh_pickups,
                'deliveries': [],
                'unloads': []
            })

            prev_node = other_wh_node

    # Step 4: Deliver split order
    split_dest = split_order.destination.id
    path_to_split_dest = dijkstra_path(adjacency_list, prev_node, split_dest)

    if path_to_split_dest:
        # Intermediate nodes
        for intermediate in path_to_split_dest[:-1]:
            steps.append({
                'node_id': intermediate,
                'pickups': [],
                'deliveries': [],
                'unloads': []
            })

        # Delivery of split order
        split_deliveries = []
        for sku_id, qty in split_order.requested_items.items():
            split_deliveries.append({
                'order_id': split_order_id,
                'sku_id': sku_id,
                'quantity': qty
            })

        steps.append({
            'node_id': split_dest,
            'pickups': [],
            'deliveries': split_deliveries,
            'unloads': []
        })

        prev_node = split_dest

    # Step 5: Return to home warehouse
    return_path = dijkstra_path(adjacency_list, prev_node, home_node)

    if return_path:
        for node in return_path[:-1]:
            steps.append({
                'node_id': node,
                'pickups': [],
                'deliveries': [],
                'unloads': []
            })

        steps.append({
            'node_id': home_node,
            'pickups': [],
            'deliveries': [],
            'unloads': []
        })
    elif prev_node != home_node:
        # Fallback: just add home node
        steps.append({
            'node_id': home_node,
            'pickups': [],
            'deliveries': [],
            'unloads': []
        })

    return steps


def build_route_with_validation(
    env,
    vehicle,
    warehouse,
    assigned_orders: List[str],
    adjacency_list: Dict
) -> List[Dict]:
    """Build route with comprehensive validation."""
    home_node = warehouse.location.id
    steps = []

    # Step 1: Pickup all items at warehouse
    pickups = []
    total_weight = 0.0
    total_volume = 0.0

    for oid in assigned_orders:
        order = env.orders[oid]
        for sku_id, qty in order.requested_items.items():
            sku = env.skus[sku_id]
            pickups.append({
                'warehouse_id': warehouse.id,
                'sku_id': sku_id,
                'quantity': qty
            })
            total_weight += sku.weight * qty
            total_volume += sku.volume * qty

    # Validate capacity before creating route
    if (total_weight > vehicle.capacity_weight * CAPACITY_SAFETY or
        total_volume > vehicle.capacity_volume * CAPACITY_SAFETY):
        return []  # Invalid: exceeds capacity

    steps.append({
        'node_id': home_node,
        'pickups': pickups,
        'deliveries': [],
        'unloads': []
    })

    # Step 2: Optimize delivery sequence using nearest neighbor
    delivery_nodes = {}
    for oid in assigned_orders:
        order = env.orders[oid]
        dest = order.destination.id
        if dest not in delivery_nodes:
            delivery_nodes[dest] = []
        delivery_nodes[dest].append(oid)

    # Nearest neighbor tour
    tour = nearest_neighbor_tour(home_node, list(delivery_nodes.keys()), adjacency_list)

    # Step 3: Build steps with intermediate nodes (REQUIRED by API)
    prev_node = home_node
    for delivery_node in tour:
        # Find path
        path = dijkstra_path(adjacency_list, prev_node, delivery_node)

        # Validate path exists
        if not path:
            continue  # No path found, skip this order

        # Add intermediate nodes (required for proper routing)
        for intermediate in path[:-1]:
            steps.append({
                'node_id': intermediate,
                'pickups': [],
                'deliveries': [],
                'unloads': []
            })

        # Add delivery step
        deliveries = []
        for oid in delivery_nodes[delivery_node]:
            order = env.orders[oid]
            for sku_id, qty in order.requested_items.items():
                deliveries.append({
                    'order_id': oid,
                    'sku_id': sku_id,
                    'quantity': qty
                })

        steps.append({
            'node_id': delivery_node,
            'pickups': [],
            'deliveries': deliveries,
            'unloads': []
        })

        prev_node = delivery_node

    # Step 4: Return home
    return_path = dijkstra_path(adjacency_list, prev_node, home_node)

    if return_path:  # Only add return if path exists
        for node in return_path[:-1]:
            steps.append({
                'node_id': node,
                'pickups': [],
                'deliveries': [],
                'unloads': []
            })

        steps.append({
            'node_id': home_node,
            'pickups': [],
            'deliveries': [],
            'unloads': []
        })
    elif prev_node == home_node:
        # Already at home
        pass
    else:
        # Cannot return home - invalid route
        return []

    return steps


def nearest_neighbor_tour(start: int, nodes: List[int], adjacency_list: Dict) -> List[int]:
    """
    Create tour using improved nearest neighbor heuristic with 2-opt optimization.

    MERGED OPTIMIZATIONS:
    1. Smart start: Begin with node closest to warehouse (reduces initial leg)
    2. Greedy nearest neighbor for remaining nodes
    3. 2-opt local search to improve tour quality
    """
    if not nodes:
        return []

    if len(nodes) == 1:
        return list(nodes)

    # Phase 1: Build initial tour with SMART nearest neighbor
    unvisited = set(nodes)

    # OPTIMIZATION: Start with the delivery closest to warehouse
    # This reduces the "first leg" distance significantly
    first_node = None
    min_start_dist = float('inf')
    for node in unvisited:
        dist = dijkstra_distance(adjacency_list, start, node)
        if dist < min_start_dist:
            min_start_dist = dist
            first_node = node

    if first_node is None:
        first_node = next(iter(unvisited))

    tour = [first_node]
    unvisited.remove(first_node)
    current = first_node

    # Continue with nearest neighbor
    while unvisited:
        nearest = None
        min_dist = float('inf')

        for node in unvisited:
            dist = dijkstra_distance(adjacency_list, current, node)
            if dist < min_dist:
                min_dist = dist
                nearest = node

        if nearest is None:
            nearest = next(iter(unvisited))

        tour.append(nearest)
        unvisited.remove(nearest)
        current = nearest

    # Phase 2: Improve tour with 2-opt (for tours with 3+ nodes)
    if len(tour) >= 3:
        tour = improve_tour_with_2opt(start, tour, adjacency_list, max_iterations=20)

    return tour


def improve_tour_with_2opt(start: int, tour: List[int], adjacency_list: Dict, max_iterations: int = 20) -> List[int]:
    """
    Improve tour using 2-opt local search.

    2-opt: Try swapping edges to reduce total distance.
    For each pair of edges (i, i+1) and (j, j+1), try reversing the segment between them.
    """
    if len(tour) < 3:
        return tour

    improved = True
    iteration = 0

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1

        for i in range(len(tour) - 2):
            for j in range(i + 2, len(tour)):
                # Current edges: (prev_i -> tour[i]) and (tour[j] -> next_j)
                prev_i = start if i == 0 else tour[i-1]
                next_j = start if j == len(tour) - 1 else tour[j+1]

                # Current distance
                current_dist = (dijkstra_distance(adjacency_list, prev_i, tour[i]) +
                               dijkstra_distance(adjacency_list, tour[j], next_j))

                # New distance if we reverse segment [i, j]
                new_dist = (dijkstra_distance(adjacency_list, prev_i, tour[j]) +
                           dijkstra_distance(adjacency_list, tour[i], next_j))

                # If improvement found, reverse the segment
                if new_dist < current_dist - 0.01:  # Small epsilon to avoid floating point issues
                    tour[i:j+1] = reversed(tour[i:j+1])
                    improved = True
                    break

            if improved:
                break

    return tour


# if __name__ == '__main__':
#     env = LogisticsEnvironment()
#     env.set_random_seed(42)
#     solution = my_solver(env)
#
#     is_valid, msg = env.validate_solution_complete(solution)
#     print(f"Valid: {is_valid}")
#     if is_valid:
#         success, _ = env.execute_solution(solution)
#         if success:
#             cost = env.calculate_solution_cost(solution)
#             print(f"Cost: ${cost:.2f}, Vehicles: {len(solution['routes'])}")
