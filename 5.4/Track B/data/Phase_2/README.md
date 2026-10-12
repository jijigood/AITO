# Track B: Telco Troubleshooting and Optimization Agentic Challenge

## **API Integration & Execution Guide**

To execute commands and retrieve the CLI output (echo) from the simulation sandbox, send an HTTP POST request to the Agent API endpoint.

### **API Endpoints**

* **Chinese Region (ELB):** https://120.46.145.77/ip/api/agent/execute  
* **Overseas Region (Hong Kong ECS):** https://124.71.227.61/ip/api/agent/execute

### **Request Headers**

Content-Type: application/json  
Authorization: Bearer \<Your-Token\>

### **JSON Payload Structure**

| Parameter | Type | Required | Description |
| :---- | :---- | :---- | :---- |
| device_name | String | Yes | Target device hostname (e.g., AGG_SW_01, BJHQ_CSR1000V_GW_01). |
| command | String | Yes | The exact CLI command to execute (must conform to the regex whitelist). |
| question_number | String | Yes | Scenario or question identifier (e.g., "34", "others"). |

### **Rate Limiting & Concurrency Rules**

To ensure fair usage and system stability, the API enforces strict limits based on your Authorization token:

* **Concurrency Limit:** Only **1 concurrent request** is allowed at any given time per token. Simultaneous requests will be rejected with a 429 Too Many Requests status.  
* **Execution Limit:** A maximum of **500 API calls** is permitted per question_number (scenario) for each token.

### **Python Usage Example**

Below is a standard Python implementation to demonstrate end-to-end API availability, showing how to send normal instructions, trigger the intelligent error-correction engine, and handle rate-limiting.

```python
import traceback  
import requests  
import json  
import time  
import urllib3

# Suppress self-signed certificate warnings for public IP testing  
urllib3.disable_warnings() 

# ELB for Chinese region  
BASE_URL = "[https://120.46.145.77/ip/api/agent/execute](https://120.46.145.77/ip/api/agent/execute)"  
# HONGKONG ECS for overseas regions  
# BASE_URL = "[https://124.71.227.61/ip/api/agent/execute](https://124.71.227.61/ip/api/agent/execute)"

# Replace with the actual player token  
HEADERS = {  
    "Content-Type": "application/json",  
    "Authorization": "Bearer ip-xxx"  
}

def run_test(scenario_name, payload):  
    print(f"\\n[{scenario_name}] sending requests...")  
    print(f"Payload: {payload}")  
    try:  
        start_time = time.time()  
        # verify=False is used to ignore HTTPS self-signed certificate warnings  
        response = requests.post(BASE_URL, headers=HEADERS, json=payload, timeout=20, verify=False)  
        latency = (time.time() - start_time) \* 1000

        print(f"✅ Status Code: {response.status_code} (Latency: {latency:.2f}ms)")  
        print(f"📦 Response Content:\\n{json.dumps(response.json(), indent=2, ensure_ascii=False)}")

    except requests.exceptions.ConnectionError:  
        print("❌ Fatal Error: Connection is refused. Please check the server status.")  
    except Exception as e:  
        print(f"❌ Request Error: {str(e)}")  
        traceback.print_exc()

if __name__ == "__main__":  
    print("==================================================")  
    print("🚀 Pre-competition API Interface End-to-End Testing")  
    print("==================================================")

    # Scenario 1: Completely correct instruction   
    # (Tests cache hit or simulated 404 missing file mechanism)  
    payload_valid = {  
        "device_name": "AGG_SW_01",  
        "command": "display current-configuration",  
        "question_number": "34"  
    }  
    run_test("Scenario 1 - Normal Query Instruction", payload_valid)

    # Scenario 2: Incorrect device instruction  
    # (Tests the intelligent error-correction engine handling invalid vendor syntax)  
    payload_syntax_error = {  
        "device_name": "AGG_SW_01",  
        "command": "display eth-trunk",  
        "question_number": "34"  
    }  
    run_test("Scenario 2 - Simulating syntax error instruction", payload_syntax_error)

    # Scenario 3: Testing concurrency limiting  
    # (Sending multiple requests instantly will trigger the rate-limiter)  
    payload_rate_limit = {  
        "device_name": "AGG_SW_01",   
        "command": "show ip int brief",  
        "question_number": "others"  
    }  
    run_test("Scenario 3 - Testing Traffic Verification", payload_rate_limit)
```
**Regarding Security and Anti-Escape:**

Before proceeding to cache matching, all the above commands undergo strict Token authentication, Redis-based concurrency lock control and rate limiting, as well as the question bank's no_permission whitelist verification. Command inputs that fail to match the aforementioned regexes will be intercepted by the underlying "CLI Error Simulation Engine" and will dynamically generate realistic, vendor-level error prompts.

## **Network Device Simulation Sandbox (Agent CLI) Supported Commands List**

This details all the device command-line interfaces (CLIs) supported in the network automation Agent sandbox engine. The current backend system performs strict validation using a **Regular Expression (Regex) whitelist**.

Supported vendors include: **Huawei**, **Cisco**, **H3C**, and **Linux (Servers)**.

**Parameter Notation Conventions:**

* \<parameter\>: Represents a mandatory variable, such as an IP address, instance name, etc. (matched by the regex \\S+).  
* [parameter]: Represents an optional parameter or keyword (defined by the regex ()?, can be omitted).

### **1. Huawei**

Huawei devices support the richest set of commands, comprehensively covering basic system information, Layer 2/Layer 3 networking, security policies, and advanced data center features.

#### **1.1 Basic Information and System Status**

* display current-configuration  
* display current-configuration | include ip route-static  
* display current-configuration | include nat  
* display logbuffer  
* display alarm active  
* display memory[-usage] *(Compatible with both memory and memory-usage formats)*

#### **1.2 Interfaces and Layer 2 Networks**

* display interface brief  
* display interface description  
* display ip interface brief  
* display eth-trunk [interface_number] *(Can be viewed globally or precisely for a specific aggregated interface)*  
* display vlan  
* display mac-address  
* display lldp neighbor [brief | verbose]  
* display stp [interface] brief

#### **1.3 ARP and IPv6**

* display arp [all]  
* display ipv6 neighbors

#### **1.4 Routing and VPN (OSPF/BGP/MPLS)**

* display ip routing-table [all-vpn-instance | vpn-instance \<instance_name\> [\<extended_parameters\>]] *(Supports appending advanced parameters for querying)*  
* display ipv6 routing-table  
* display ip vpn-instance [\<instance_name\> [verbose]]  
* display ospf peer  
* display ospf routing  
* display ospf interface  
* display ospf lsdb  
* display bgp peer  
* display bgp routing-table  
* display bgp evpn all routing-table  
* display bgp vpnv4 all routing-table [verbose]  
* display mpls lsp

#### **1.5 Advanced Features (QoS/VXLAN/SRv6/VRRP/BFD/DHCP)**

* display traffic-policy *(QoS/Traffic Policy)*  
* display vxlan tunnel  
* display vxlan troubleshooting  
* display vrrp verbose  
* display bfd session all  
* display ip pool  
* display srv6-te policy [status]  
* display segment-routing ipv6 local-sid end forwarding  
* display segment-routing ipv6 local-sid end-x forwarding

#### **1.6 Firewall and Security Policies (USG Features)**

* display acl all  
* display security-policy  
* display zone  
* display firewall session table  
* display nat policy  
* display nat session

### **2. Cisco**

Cisco devices mainly support standard IOS/NX-OS daily operations and troubleshooting commands.

#### **2.1 Basic Information and System Status**

* show running-config  
* show logging  
* show facility-alarm status  
* show processes memory

#### **2.2 Interfaces and Layer 2 Networks**

* show interface brief or show ip interface brief  
* show interface description  
* show ipv6 interface [\<parameter1\> \<parameter2\>] *(e.g., show ipv6 interface brief or show ipv6 interface GigabitEthernet0/0/0 brief)*  
* show port-channel summary or show etherchannel summary  
* show vlan or show vlans  
* show mac address-table  
* show lldp neighbors  
* show spanning-tree summary or show spanning-tree brief

#### **2.3 ARP and IPv6**

* show ip arp  
* show ipv[6] neighbor[s] *(Compatible with various abbreviations omitting 6 and s)*

#### **2.4 Routing and VPN**

* show ip route  
* show ipv6 route  
* show ip route vrf \<instance_name\>  
* show ip ospf neighbor[s]  
* show ip route ospf  
* show ip ospf database  
* show bgp l2vpn evpn  
* show bgp vpnv4 unicast all

#### **2.5 Advanced Features**

* show nve vni  
* show nve peers  
* show nve.\* *(Super wildcard: Allows all subsequent commands starting with show nve)*  
* show vrrp detail  
* show bfd neighbors  
* show ip dhcp pool  
* show segment-routing srv6 policy

### **3. H3C**

The command structure of H3C is very similar to Huawei's, but differs in specific keywords (such as aggregated interfaces, neighbors, etc.).

#### **3.1 Basic Information and Interfaces**

* display current-configuration  
* display logbuffer  
* display alarm [active]  
* display memory  
* display interface brief  
* display ip interface brief  
* display link-aggregation summary *(Corresponds to Huawei's eth-trunk)*  
* display vlan  
* display mac-address  
* display lldp neighbor-information  
* display stp [interface] brief

#### **3.2 ARP and IPv6**

* display arp all  
* display ipv6 neighbor / display ipv6 neighbors / display ipv6 neighbor all

#### **3.3 Routing and Advanced Features**

* display ip routing-table  
* display ipv6 routing-table  
* display ospf peer  
* display ospf routing  
* display ospf lsdb  
* display bgp l2vpn evpn  
* display bgp vpnv4 all routing-table or display bgp routing-table vpnv4 *(Supports two parameter word orders)*  
* display vxlan tunnel  
* display vxlan troubleshooting  
* display vrrp verbose  
* display bfd session  
* display ip pool  
* display segment-routing ipv6 te policy  
* display segment-routing ipv6 local-sid

### **4. Linux (Servers)**

For Linux host or virtual machine nodes, the system supports basic network configuration surveys.

* ip addr  
* ip neigh show dev eth0  
* ip route [show] *(Supports ip route or ip route show)*  
* ifconfig [\<interface_name\>] *(Supports global view or specifying an interface, e.g., ifconfig eth0)*
