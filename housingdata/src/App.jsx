import { useState } from 'react'


const App = () => {

  const [postalCode, setPostalCode] = useState('')
  const [livingSpace, setLivingSpace] = useState('')
  const [apartmentPrice, setApartmentPrice] = useState('')

  const handlePostalCodeChange = (event) => {
    setPostalCode(event.target.value)
  }

  const handleLivingSpaceChange = (event) => {
    setLivingSpace(event.target.value)
  }

  const handleApartmentPriceChange = (event) => {
    setApartmentPrice(event.target.value)
  }

  /*
  Get values from forms and send to backend
  */
  const getApartments = (event) => {
    event.preventDefault()
    fetch('http://127.0.0.1:5000/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        postalCode: postalCode,
        livingSpace: livingSpace,
        apartmentPrice: apartmentPrice
    })
    });
  }

  return (
    <div>
      <h1>Housing Data Analyzer</h1>
      <form onSubmit={getApartments}>
        Postal Code:<br/>
        <input value={postalCode} onChange={handlePostalCodeChange}/><br/>
        Living Space:<br/>
        <input value={livingSpace} onChange={handleLivingSpaceChange}/><br/>
        Apartment Price:<br/>
        <input value={apartmentPrice} onChange={handleApartmentPriceChange}/><br/>
        <button type="submit">Submit</button>
      </form>
    </div>
  )
}

export default App
