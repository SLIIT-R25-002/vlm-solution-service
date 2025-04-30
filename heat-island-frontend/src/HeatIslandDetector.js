import React, { useState } from 'react';
import { Container, Row, Col, Form, Button, Table, Alert, Card } from 'react-bootstrap';
import axios from 'axios';

const HeatIslandDetector = () => {
  const [data, setData] = useState([]);
  const [newEntry, setNewEntry] = useState({
    locationType: '',
    material: '',
    temperature: '',
    humidity: '',
    area: ''
  });
  const [editingIndex, setEditingIndex] = useState(null);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const materialOptions = [
    "asphalt", "concrete", "grass", "metal", "plastic",
    "rubber", "sand", "soil", "solar panel", "steel",
    "water", "artificial turf", "glass"
  ];

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setNewEntry(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const addOrUpdateDataEntry = () => {
    if (!newEntry.locationType || !newEntry.material ||
      !newEntry.temperature || !newEntry.humidity || !newEntry.area) {
      alert("Please fill in all fields");
      return;
    }

    const entry = [
      newEntry.locationType,
      newEntry.material,
      parseFloat(newEntry.temperature),
      parseFloat(newEntry.humidity),
      parseFloat(newEntry.area)
    ];

    if (editingIndex !== null) {
      console.log(`[UPDATE] Editing index ${editingIndex}:`, entry);
      const updatedData = [...data];
      updatedData[editingIndex] = entry;
      setData(updatedData);
      setEditingIndex(null);
    } else {
      console.log("[ADD] New entry:", entry);
      setData(prev => [...prev, entry]);
    }

    setNewEntry({
      locationType: '',
      material: '',
      temperature: '',
      humidity: '',
      area: ''
    });
  };

  const startEditEntry = (index) => {
    console.log(`[EDIT] Index: ${index}`);
    const entryToEdit = data[index];
    setNewEntry({
      locationType: entryToEdit[0],
      material: entryToEdit[1],
      temperature: entryToEdit[2].toString(),
      humidity: entryToEdit[3].toString(),
      area: entryToEdit[4].toString()
    });
    setEditingIndex(index);
  };

  const removeDataEntry = (indexToRemove) => {
    console.log(`[REMOVE] Index: ${indexToRemove}`);
    setData(prev => prev.filter((_, index) => index !== indexToRemove));
  };

  const predictHeatIsland = async () => {
    if (data.length === 0) {
      alert("Please add at least one data entry");
      return;
    }

    try {
      setLoading(true);
      setError(null);
      console.log("[REQUEST] Sending data to backend:", data);

      const response = await axios.post('http://localhost:5000/predict', { data });

      console.log("[RESPONSE] Received:", response.data);
      setResults(response.data);
    } catch (err) {
      console.error("[ERROR] Prediction failed:", err.response?.data?.error || err.message);
      setError(err.response?.data?.error || "An error occurred");
      setResults(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container className="mt-5">
      <h1 className="text-center mb-4">🔥 Urban Heat Island Detector 🔥</h1>

      <Card className="mb-4">
        <Card.Header>
          {editingIndex !== null ? 'Edit Data Entry' : 'Add New Data Entry'}
        </Card.Header>
        <Card.Body>
          <Form>
            <Row>
              <Col md={4}>
                <Form.Group className="mb-3">
                  <Form.Label>Location Type</Form.Label>
                  <Form.Control
                    type="text"
                    name="locationType"
                    value={newEntry.locationType}
                    onChange={handleInputChange}
                    placeholder="Enter location type"
                  />
                </Form.Group>
              </Col>
              <Col md={4}>
                <Form.Group className="mb-3">
                  <Form.Label>Material</Form.Label>
                  <Form.Select
                    name="material"
                    value={newEntry.material}
                    onChange={handleInputChange}
                  >
                    <option value="">Select Material</option>
                    {materialOptions.map(material => (
                      <option key={material} value={material}>{material}</option>
                    ))}
                  </Form.Select>
                </Form.Group>
              </Col>
              <Col md={4}>
                <Form.Group className="mb-3">
                  <Form.Label>Temperature (°C)</Form.Label>
                  <Form.Control
                    type="number"
                    name="temperature"
                    value={newEntry.temperature}
                    onChange={handleInputChange}
                    placeholder="Enter temperature"
                  />
                </Form.Group>
              </Col>
            </Row>
            <Row>
              <Col md={4}>
                <Form.Group className="mb-3">
                  <Form.Label>Humidity (%)</Form.Label>
                  <Form.Control
                    type="number"
                    name="humidity"
                    value={newEntry.humidity}
                    onChange={handleInputChange}
                    placeholder="Enter humidity"
                  />
                </Form.Group>
              </Col>
              <Col md={4}>
                <Form.Group className="mb-3">
                  <Form.Label>Area (sq. meters)</Form.Label>
                  <Form.Control
                    type="number"
                    name="area"
                    value={newEntry.area}
                    onChange={handleInputChange}
                    placeholder="Enter area"
                  />
                </Form.Group>
              </Col>
              <Col md={4} className="d-flex align-items-end">
                <Button
                  variant={editingIndex !== null ? "primary" : "success"}
                  onClick={addOrUpdateDataEntry}
                  className="me-2"
                >
                  {editingIndex !== null ? 'Update Entry' : 'Add Entry'}
                </Button>
                {editingIndex !== null && (
                  <Button
                    variant="secondary"
                    onClick={() => {
                      setEditingIndex(null);
                      setNewEntry({
                        locationType: '',
                        material: '',
                        temperature: '',
                        humidity: '',
                        area: ''
                      });
                    }}
                  >
                    Cancel
                  </Button>
                )}
              </Col>
            </Row>
          </Form>
        </Card.Body>
      </Card>

      {data.length > 0 && (
        <Card className="mb-4">
          <Card.Header>Current Data Entries</Card.Header>
          <Card.Body>
            <Table striped bordered hover responsive>
              <thead>
                <tr>
                  <th>Location</th>
                  <th>Material</th>
                  <th>Temp (°C)</th>
                  <th>Humidity (%)</th>
                  <th>Area (sq.m)</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {data.map((entry, index) => (
                  <tr key={index}>
                    <td>{entry[0]}</td>
                    <td>{entry[1]}</td>
                    <td>{entry[2]}</td>
                    <td>{entry[3]}</td>
                    <td>{entry[4]}</td>
                    <td>
                      <Button
                        variant="outline-primary"
                        size="sm"
                        className="me-2"
                        onClick={() => startEditEntry(index)}
                      >
                        Edit
                      </Button>
                      <Button
                        variant="outline-danger"
                        size="sm"
                        onClick={() => removeDataEntry(index)}
                      >
                        Remove
                      </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </Table>
            <Button variant="success" onClick={predictHeatIsland}>
              Predict Heat Island Effect
            </Button>
          </Card.Body>
        </Card>
      )}

      {loading && (
        <div className="text-center my-4">
          <div className="spinner-border text-primary" role="status">
            <span className="visually-hidden">Loading...</span>
          </div>
          <p className="mt-2">Analyzing data... please wait</p>
        </div>
      )}

      {results && !loading && (
        <Card>
          <Card.Header>Heat Island Detection Summary</Card.Header>
          <Card.Body>
            <Row>
              <Col md={6}>
                <Card className="mb-4">
                  <Card.Header>Location Details</Card.Header>
                  <Card.Body>
                    {results.detailed_results.map((item, index) => (
                      <div
                        key={index}
                        className={`mb-2 ${item.heat_island === 'Yes' ? 'text-danger' : 'text-success'}`}
                      >
                        <strong>{item.location}</strong>
                        <span className="ms-2 badge bg-secondary">
                          {item.heat_island === 'Yes' ? 'Heat Island' : 'No Heat Island'}
                        </span>
                        <div className="small text-muted">
                          Material: {item.material} | Temp: {item.temperature}°C | Humidity: {item.humidity}% | Area: {item.area} m²
                        </div>
                      </div>
                    ))}
                  </Card.Body>
                </Card>
              </Col>
              <Col md={6}>
                <Card className="mb-4">
                  <Card.Header>Area-Based Analysis</Card.Header>
                  <Card.Body>
                    <div className="d-flex justify-content-between mb-2">
                      <span>🔥 Heat-Retaining Material:</span>
                      <strong>{results.summary.heat_retaining_percent}%</strong>
                    </div>
                    <div className="d-flex justify-content-between mb-2">
                      <span>🌿 Vegetation Coverage:</span>
                      <strong>{results.summary.vegetation_percent}%</strong>
                    </div>
                    <div className="d-flex justify-content-between mb-2">
                      <span>🌡 Avg Temperature:</span>
                      <strong>{results.summary.avg_temperature}°C</strong>
                    </div>
                    <div className="d-flex justify-content-between mb-3">
                      <span>💧 Avg Humidity:</span>
                      <strong>{results.summary.avg_humidity}%</strong>
                    </div>
                    <Alert variant={results.summary.final_decision === "Heat Island Detected" ? "danger" : "success"}>
                      <strong>Final Decision:</strong> {results.summary.final_decision}
                    </Alert>
                  </Card.Body>
                </Card>
              </Col>
            </Row>
          </Card.Body>
        </Card>
      )}

      {error && (
        <Alert variant="danger" className="mt-4">
          {error}
        </Alert>
      )}
    </Container>
  );
};

export default HeatIslandDetector;
