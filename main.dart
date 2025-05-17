import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() => runApp(MyApp());

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Local LLM Chat',
      theme: ThemeData(primarySwatch: Colors.indigo),
      home: const LLMChatPage(),
    );
  }
}



class LLMChatPage extends StatefulWidget {
  const LLMChatPage({super.key});
  @override
_LLMChatPageState createState() => _LLMChatPageState();
}

class _LLMChatPageState extends State<LLMChatPage> {
  final TextEditingController _controller = TextEditingController();
  String _responseText = "";



Future<void> generateText(String prompt) async {
  final apiUrl = "http://10.10.89.241:8000/generate";

  try {
    final response = await http.post(
     Uri.parse(apiUrl),
      headers: {"Content-Type": "application/json"},
      body: jsonEncode({"text": prompt}),
    );

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      print("Response: ${data['generated_text']}"); // ✅ should print "Hello World"
    } else {
      print("Error: ${response.statusCode}");
    }
  } catch (e) {
    print("Exception: $e");
  }
}

//   Future<void> generateText(String prompt) async {
//     final apiUrl = Uri.parse("http://127.0.0.1:8000/generate");
//    try{
//     final response = await http.post(
//       apiUrl,
//       headers: {"Content-Type": "application/json"},
//       body: jsonEncode({"text": prompt}),//,"max_lenght":50
//     );

//     if (response.statusCode == 200) {
//       final data = jsonDecode(response.body);
//       setState(() => _responseText =   data["generate_text"]);
//     } else {
//       setState(() =>  _responseText = "Error: ${response.body}");
//     }
//   }
//   catch(e){
//    setState(() {
//      _responseText = "Error:$e";
//    });
//   }
// }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("Talk to LLM!")),
      body: Padding(
        padding: EdgeInsets.all(16),
        child: Column(
          children: [
            TextField(controller: _controller,
            decoration :const InputDecoration(labelText: "Enter a prompt"),
            ),
            ElevatedButton(
              onPressed: () => generateText(_controller.text),
           
            
            child: const Text('Generate'),),
            SizedBox(height: 20), 
            Text(_responseText),
          ],
        ),
      ),
    );
  }
}

















