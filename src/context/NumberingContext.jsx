import { createContext, useContext, useRef, useState, useCallback } from 'react'

const NumberingContext = createContext()

export function NumberingProvider({ children }) {
  const figureRegistry = useRef(new Map())
  const tableRegistry = useRef(new Map())
  const [labels, setLabels] = useState({})

  const registerFigure = useCallback((key) => {
    const reg = figureRegistry.current
    if (!reg.has(key)) {
      reg.set(key, reg.size + 1)
    }
    return reg.get(key)
  }, [])

  const registerTable = useCallback((key) => {
    const reg = tableRegistry.current
    if (!reg.has(key)) {
      reg.set(key, reg.size + 1)
    }
    return reg.get(key)
  }, [])

  const registerLabel = useCallback((label, type, number) => {
    if (!label) return
    setLabels((prev) => {
      if (prev[label]?.type === type && prev[label]?.number === number) return prev
      return { ...prev, [label]: { type, number } }
    })
  }, [])

  const getLabel = useCallback((label) => {
    return labels[label] || null
  }, [labels])

  return (
    <NumberingContext.Provider
      value={{ registerFigure, registerTable, registerLabel, getLabel }}
    >
      {children}
    </NumberingContext.Provider>
  )
}

export function useNumbering() {
  const ctx = useContext(NumberingContext)
  if (!ctx) throw new Error('useNumbering must be used within NumberingProvider')
  return ctx
}
