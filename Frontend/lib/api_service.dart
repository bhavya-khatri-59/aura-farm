import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:location/location.dart';

class ApiService {
  // Dynamic IP selection based on device type
  static String get baseUrl {
    // For testing, you can manually set your computer's IP here
    // Replace 'YOUR_COMPUTER_IP' with your actual IP (e.g., '192.168.1.100')
    const String computerIP = 'YOUR_COMPUTER_IP'; // Update this!

    // Use 10.0.2.2 for emulator, computer IP for physical device
    // You can also just use computerIP for both if you know your IP
    return 'http://10.0.2.2:8000'; // Change to http://$computerIP:8000 for physical device
  }

  static Future<String> diagnosePlant({
    required String message,
    File? image,
  }) async {
    try {
      // Get current location
      Location location = Location();
      bool serviceEnabled;
      PermissionStatus permissionGranted;
      LocationData locationData;

      // Check if location service is enabled
      serviceEnabled = await location.serviceEnabled();
      if (!serviceEnabled) {
        serviceEnabled = await location.requestService();
        if (!serviceEnabled) {
          return 'Location service is disabled. Please enable location services.';
        }
      }

      // Check location permissions
      permissionGranted = await location.hasPermission();
      if (permissionGranted == PermissionStatus.denied) {
        permissionGranted = await location.requestPermission();
        if (permissionGranted != PermissionStatus.granted) {
          return 'Location permission denied. Using default location.';
        }
      }

      // Get current location
      try {
        locationData = await location.getLocation().timeout(
          Duration(seconds: 10),
          onTimeout: () {
            throw Exception('Location timeout');
          },
        );
      } catch (e) {
        // If location fails, use default coordinates
        print('Failed to get location: $e');
        locationData = LocationData.fromMap({
          'latitude': 28.6139,
          'longitude': 77.2090,
        });
      }

      var request = http.MultipartRequest(
        'POST',
        Uri.parse('$baseUrl/diagnose'),
      );

      // Add text prompt
      request.fields['prompt'] = message;

      // Add current location coordinates
      request.fields['lat'] = locationData.latitude?.toString() ?? '28.6139';
      request.fields['lon'] = locationData.longitude?.toString() ?? '77.2090';

      print(
        'Using coordinates: ${locationData.latitude}, ${locationData.longitude}',
      );

      // Add image if provided
      if (image != null) {
        try {
          // Check if file exists before trying to read it
          if (!await image.exists()) {
            return 'Error: Image file no longer exists. Please select the image again.';
          }

          // Get file length safely
          int length;
          try {
            length = await image.length();
          } catch (e) {
            return 'Error: Cannot read image file. Please select the image again.';
          }

          if (length == 0) {
            return 'Error: Image file is empty. Please select a different image.';
          }

          var stream = http.ByteStream(image.openRead());
          var multipartFile = http.MultipartFile(
            'image',
            stream,
            length,
            filename: 'plant_image.jpg',
          );
          request.files.add(multipartFile);
          print('Image added to request successfully (${length} bytes)');
        } catch (e) {
          return 'Error reading image file: $e. Please select the image again.';
        }
      }

      // Set headers
      request.headers.addAll({'Content-Type': 'multipart/form-data'});

      // Send request
      var response = await request.send();
      var responseData = await response.stream.bytesToString();

      if (response.statusCode == 200) {
        var jsonResponse = json.decode(responseData);
        return jsonResponse['response'] ?? 'No response from server';
      } else {
        return 'Error: ${response.statusCode} - $responseData';
      }
    } catch (e) {
      return 'Network error: $e';
    }
  }

  static Future<String> sendChatHistory(
    List<Map<String, dynamic>> history,
  ) async {
    try {
      final url = Uri.parse('$baseUrl/history');

      final response = await http.post(
        url,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'history': history}),
      );

      if (response.statusCode == 200) {
        final responseData = jsonDecode(response.body);
        return responseData['message'] ?? 'History sent successfully';
      } else {
        final responseData = jsonDecode(response.body);
        return 'Error: ${response.statusCode} - $responseData';
      }
    } catch (e) {
      return 'Network error: $e';
    }
  }
}
