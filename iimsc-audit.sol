// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract FISIE {
    struct IoTDevice {
        address addr;
        uint funds;
        bool registered;
    }

    struct FogNode {
        address addr;
        uint collateral;
        uint reputation;
        bool registered;
    }

    address public oracle;
    mapping(address => IoTDevice) public iotDevices;
    mapping(address => FogNode) public fogNodes;
    uint public constant INIT_REPUTATION = 50;
    uint public constant MAX_REPUTATION = 100;
    uint public constant MIN_REPUTATION = 10;
    uint public constant PENALTY = 10;
    uint public constant REWARD = 5;

    event IoTRegistered(address indexed device);
    event FogRegistered(address indexed node);
    event Payment(address indexed device, address indexed node, uint amount);
    event AuditResult(address indexed node, bool success);

    modifier onlyOracle() {
        require(msg.sender == oracle, "Not authorized");
        _;
    }

    constructor() {
        oracle = msg.sender;
    }

    function registerIoT(address device) external {
        require(!iotDevices[device].registered, "Already registered");
        iotDevices[device] = IoTDevice(device, 0, true);
        emit IoTRegistered(device);
    }

    function registerFog(address node, uint collateral) external {
        require(!fogNodes[node].registered, "Already registered");
        require(collateral > 0, "Invalid collateral");
        fogNodes[node] = FogNode(node, collateral, INIT_REPUTATION, true);
        emit FogRegistered(node);
    }

    function depositFunds(address device) external payable {
        require(iotDevices[device].registered, "Device not registered");
        iotDevices[device].funds += msg.value;
    }

    function requestService(address node, uint payment) external {
        require(iotDevices[msg.sender].registered, "Device not registered");
        require(fogNodes[node].registered, "Fog node not registered");
        require(iotDevices[msg.sender].funds >= payment, "Insufficient funds");
        
        iotDevices[msg.sender].funds -= payment;
        payable(node).transfer(payment);
        emit Payment(msg.sender, node, payment);
    }

    function auditResult(address node, bool success) external onlyOracle {
        require(fogNodes[node].registered, "Fog node not registered");
        if (success) {
            fogNodes[node].reputation = min(fogNodes[node].reputation + REWARD, MAX_REPUTATION);
        } else {
            fogNodes[node].reputation = max(fogNodes[node].reputation - PENALTY, MIN_REPUTATION);
            fogNodes[node].collateral -= PENALTY;
        }
        emit AuditResult(node, success);
    }

    function min(uint a, uint b) internal pure returns (uint) {
        return a < b ? a : b;
    }

    function max(uint a, uint b) internal pure returns (uint) {
        return a > b ? a : b;
    }
}
