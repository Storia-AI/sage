import React, { useState } from 'react';

// Define the types for the props
interface MyComponentProps {
  title: string;
  subtitle?: string; // Optional prop
}

const MyComponent: React.FC<MyComponentProps> = ({ title, subtitle }) => {
  // Define a state variable with an initial value
  const [count, setCount] = useState<number>(0);

  // Function to handle button click
  const handleButtonClick = () => {
    setCount(count + 1);
  };

  return (
    <div style={{ padding: '20px', border: '1px solid #ccc', borderRadius: '8px' }}>
      <h1>{title}</h1>
      {subtitle && <h2>{subtitle}</h2>}
      <p>Current count: {count}</p>
      <button onClick={handleButtonClick}>Increase count</button>
    </div>
  );
};

export default MyComponent;
