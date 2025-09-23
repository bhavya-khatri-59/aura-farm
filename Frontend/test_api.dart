import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

// Simple test script to verify API response
void main() async {
  await testApiConnection();
}

Future<void> testApiConnection() async {
  try {
    print('Testing API connection to http://127.0.0.1:8000/diagnose');

    var request = http.MultipartRequest(
      'POST',
      Uri.parse('http://127.0.0.1:8000/diagnose'),
    );

    // Add test data
    request.fields['prompt'] = 'Test message: What is this plant?';
    request.fields['lat'] = '28.6139';
    request.fields['lon'] = '77.2090';

    // Try to add a test image (using the assets bg.png for testing)
    try {
      File testImage = File('assets/bg.png');
      if (await testImage.exists()) {
        var stream = http.ByteStream(testImage.openRead());
        var length = await testImage.length();
        var multipartFile = http.MultipartFile(
          'image',
          stream,
          length,
          filename: 'test_plant.jpg',
        );
        request.files.add(multipartFile);
        print('Added test image to request');
      } else {
        print('Test image not found - sending request without image');
      }
    } catch (e) {
      print('Could not add image: $e');
    }

    print('Sending request...');
    var response = await request.send();
    var responseData = await response.stream.bytesToString();

    print('Status Code: ${response.statusCode}');
    print('Response: $responseData');

    if (response.statusCode == 200) {
      var jsonResponse = json.decode(responseData);
      print('Parsed Response: ${jsonResponse['response']}');
    }
  } catch (e) {
    print('Error: $e');
  }
}
