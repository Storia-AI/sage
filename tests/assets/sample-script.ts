function bubbleSort(arr: number[]): number[] {
    let n = arr.length;
    let swapped: boolean;

    // Outer loop for traversing the array
    for (let i = 0; i < n; i++) {
        swapped = false;

        // Inner loop for comparing adjacent elements
        for (let j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                // Swap the elements if they are in the wrong order
                let temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
                swapped = true;
            }
        }

        // If no elements were swapped in the inner loop, break out of the loop
        if (!swapped) {
            break;
        }
    }

    return arr;
}

// Example usage
let arr = [64, 34, 25, 12, 22, 11, 90];
console.log("Sorted array:", bubbleSort(arr));

function mergeSort(arr: number[]): number[] {
    // Base case: if the array has only one element or is empty, return it
    if (arr.length <= 1) {
        return arr;
    }

    // Find the middle point of the array
    const middle = Math.floor(arr.length / 2);

    // Divide the array into left and right halves
    const left = arr.slice(0, middle);
    const right = arr.slice(middle);

    // Recursively sort both halves and then merge them
    return merge(mergeSort(left), mergeSort(right));
}

// Function to merge two sorted arrays
function merge(left: number[], right: number[]): number[] {
    let resultArray: number[] = [];
    let leftIndex = 0;
    let rightIndex = 0;

    // Compare the elements in the left and right arrays and merge them in sorted order
    while (leftIndex < left.length && rightIndex < right.length) {
        if (left[leftIndex] < right[rightIndex]) {
            resultArray.push(left[leftIndex]);
            leftIndex++;
        } else {
            resultArray.push(right[rightIndex]);
            rightIndex++;
        }
    }

    // Concatenate any remaining elements in the left or right arrays
    return resultArray.concat(left.slice(leftIndex)).concat(right.slice(rightIndex));
}

// Example usage
let arr2 = [38, 27, 43, 3, 9, 82, 10];
console.log("Sorted array:", mergeSort(arr2));

function quickSort(arr: number[]): number[] {
    // Base case: arrays with 0 or 1 element are already sorted
    if (arr.length <= 1) {
        return arr;
    }

    // Choose a pivot element from the array
    const pivot = arr[arr.length - 1];

    // Arrays to hold elements less than, equal to, and greater than the pivot
    const left: number[] = [];
    const right: number[] = [];
    const equal: number[] = [];

    // Partition the array into left (less than pivot), equal (equal to pivot), and right (greater than pivot)
    for (let num of arr) {
        if (num < pivot) {
            left.push(num);
        } else if (num > pivot) {
            right.push(num);
        } else {
            equal.push(num);
        }
    }

    // Recursively sort left and right, then concatenate the result
    return [...quickSort(left), ...equal, ...quickSort(right)];
}

// Example usage
let arr3 = [29, 10, 14, 37, 13];
console.log("Sorted array:", quickSort(arr3));

