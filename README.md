First Multi-Object-Tracking practice: Kalman Filter

to use the kalman filter class, simply instantiate an instance of it and call that instance each time a new measurement is taken - use these measurements as the args. This will return the updated predicted state, from which you can take the required values.