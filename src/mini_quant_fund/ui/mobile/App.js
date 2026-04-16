import React, { useState, useEffect } from 'react';
import { StyleSheet, Text, View, ScrollView, TouchableOpacity, SafeAreaView, StatusBar } from 'react-native';

/**
 * Mini-Quant Fund: Mobile Trading Terminal Scaffold
 * 
 * Provides a React Native interface for:
 * - Real-time portfolio monitoring
 * - Manual trade execution
 * - AI strategy alerts
 * - Copy-trading management
 */

const TradingTerminal = () => {
  const [balance, setBalance] = useState(1250000.42);
  const [pnl, setPnl] = useState(15230.12);
  const [positions, setPositions] = useState([
    { symbol: 'AAPL', qty: 150, price: 182.41, change: 1.2 },
    { symbol: 'TSLA', qty: 50, price: 175.22, change: -0.8 },
    { symbol: 'NVDA', qty: 200, price: 890.11, change: 4.5 },
  ]);

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" />
      
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Mini-Quant Terminal</Text>
        <Text style={styles.balance}>${balance.toLocaleString()}</Text>
        <Text style={[styles.pnl, pnl >= 0 ? styles.positive : styles.negative]}>
          {pnl >= 0 ? '+' : ''}${pnl.toLocaleString()} (Today)
        </Text>
      </View>

      {/* Main Terminal View */}
      <ScrollView style={styles.content}>
        <Text style={styles.sectionTitle}>Active Positions</Text>
        {positions.map((pos) => (
          <View key={pos.symbol} style={styles.positionRow}>
            <View>
              <Text style={styles.symbolText}>{pos.symbol}</Text>
              <Text style={styles.qtyText}>{pos.qty} Shares</Text>
            </View>
            <View style={styles.priceContainer}>
              <Text style={styles.priceText}>${pos.price}</Text>
              <Text style={[styles.changeText, pos.change >= 0 ? styles.positive : styles.negative]}>
                {pos.change >= 0 ? '▲' : '▼'} {Math.abs(pos.change)}%
              </Text>
            </View>
          </View>
        ))}

        <View style={styles.buttonContainer}>
          <TouchableOpacity style={[styles.button, styles.buyButton]}>
            <Text style={styles.buttonText}>Quick Buy</Text>
          </TouchableOpacity>
          <TouchableOpacity style={[styles.button, styles.sellButton]}>
            <Text style={styles.buttonText}>Quick Sell</Text>
          </TouchableOpacity>
        </View>

        <Text style={styles.sectionTitle}>AI Strategy Alerts</Text>
        <View style={styles.alertCard}>
          <Text style={styles.alertTitle}>Gamma Squeeze Detected (NVDA)</Text>
          <Text style={styles.alertBody}>Execution engine increasing participation rate to 15%.</Text>
        </View>
      </ScrollView>

      {/* Tab Bar Mock */}
      <View style={styles.tabBar}>
        <Text style={styles.tabItemActive}>Trade</Text>
        <Text style={styles.tabItem}>Strategies</Text>
        <Text style={styles.tabItem}>Copy</Text>
        <Text style={styles.tabItem}>Profile</Text>
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0a0a0a',
  },
  header: {
    padding: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#333',
    alignItems: 'center',
  },
  headerTitle: {
    color: '#888',
    fontSize: 14,
    textTransform: 'uppercase',
    letterSpacing: 2,
  },
  balance: {
    color: '#fff',
    fontSize: 36,
    fontWeight: 'bold',
    marginVertical: 5,
  },
  pnl: {
    fontSize: 18,
    fontWeight: '600',
  },
  content: {
    flex: 1,
    padding: 15,
  },
  sectionTitle: {
    color: '#fff',
    fontSize: 20,
    fontWeight: 'bold',
    marginTop: 20,
    marginBottom: 10,
  },
  positionRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    backgroundColor: '#1a1a1a',
    padding: 15,
    borderRadius: 10,
    marginBottom: 10,
  },
  symbolText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  qtyText: {
    color: '#888',
    fontSize: 14,
  },
  priceContainer: {
    alignItems: 'flex-end',
  },
  priceText: {
    color: '#fff',
    fontSize: 18,
  },
  changeText: {
    fontSize: 14,
  },
  positive: {
    color: '#4caf50',
  },
  negative: {
    color: '#f44336',
  },
  buttonContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 20,
  },
  button: {
    flex: 1,
    padding: 15,
    borderRadius: 8,
    alignItems: 'center',
    marginHorizontal: 5,
  },
  buyButton: {
    backgroundColor: '#4caf50',
  },
  sellButton: {
    backgroundColor: '#f44336',
  },
  buttonText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 16,
  },
  alertCard: {
    backgroundColor: '#1a1a1a',
    borderLeftWidth: 4,
    borderLeftColor: '#2196f3',
    padding: 15,
    borderRadius: 5,
    marginTop: 10,
  },
  alertTitle: {
    color: '#2196f3',
    fontWeight: 'bold',
    marginBottom: 5,
  },
  alertBody: {
    color: '#ccc',
  },
  tabBar: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    paddingVertical: 15,
    borderTopWidth: 1,
    borderTopColor: '#333',
    backgroundColor: '#000',
  },
  tabItem: {
    color: '#888',
  },
  tabItemActive: {
    color: '#fff',
    fontWeight: 'bold',
  },
});

export default TradingTerminal;
