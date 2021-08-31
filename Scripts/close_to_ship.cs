using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class close_to_ship : MonoBehaviour
{
    public GameObject ship_object;
    public GameObject cube;
    public float threshold_distance;
    public UnityEngine.UI.Text outText;

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        Vector3 distance_vector = ship_object.transform.position - cube.transform.position;
        float distance = distance_vector.magnitude;

        if (distance < threshold_distance){
            cube.SetActive(false);
            outText.text = "Cargo retrieved.";
        }
    }
}
